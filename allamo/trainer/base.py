import gc
import os
import time
import math
import datetime
import pprint
import subprocess
import torch
import torch.distributed as dist
import torch.nn.functional as F
from contextlib import nullcontext
from dataclasses import dataclass

from allamo.checkpoint.checkpoint_manager import CheckpointManager
from allamo.configuration import AllamoConfiguration
from allamo.model.modeling_utils import get_model_spec, BaseModel
from allamo.dataset.data_loader import AllamoDataLoader
from allamo.metrics.metrics_logger import MetricsLogger
from allamo.logging import configure_logger, logger
from allamo.model.attentions import attention_version
from allamo.optimizer.optimizer_utils import calculate_learning_rate
from allamo.parallelisms.fsdp2_utils import build_world_mesh
from allamo.torch_utils import init_torch
from allamo.train_utils import (
    format_seconds_as_time,
    estimate_mfu,
    get_model_checkpoint_path,
    get_config_checkpoint_path,
    create_model_config,
    get_log_prob,
)
from allamo.training_context import TrainingContext

@dataclass
class ModelOutput:
    loss: torch.Tensor
    accuracy: float
    unmasked_labels: int
    draft_loss: torch.Tensor = None
    draft_accuracy: float = 0.0
    draft_unmasked_labels: int = 0
    draft_ignored_groups: int = 0
    draft_accepted_groups: int = 0

class BaseTrainer:
    
    def __init__(self, config: AllamoConfiguration, train_ctx: TrainingContext):       
        self.config = config
        self.train_ctx = train_ctx
        self.init_torch()
        logger.info(f"Torch initialized for run {self.train_ctx.run_uuid}")
        
        # DCP activates FSDP2
        if self.distributed() and self.config.distributed_checkpoint:
            assert self.config.dtype != 'float16', "GradScaler is not functioning properly with FSDP2"
            self.world_mesh = build_world_mesh(self.train_ctx, self.device_type)
        else:
            self.world_mesh = None
        
        if self.world_mesh is None:
            self.dp_world_size = self.train_ctx.world_size
            self.dp_rank = self.train_ctx.rank
            self.tp_rank = 0
        else:
            dp_mesh = self.world_mesh["dp"]
            tp_mesh = self.world_mesh["tp"]
            self.dp_world_size = dp_mesh.size()
            self.dp_rank = dp_mesh.get_local_rank()
            self.tp_rank = tp_mesh.get_local_rank()
        
        self.init_dataloaders()
        self.init_training()
        self.init_metrics_logger()

    def distributed(self):
        raise NotImplementedError("Not implemented")

    def init_torch(self):
        self.device_type = 'cuda' if 'cuda' in self.config.device else 'cpu'
        init_torch(self.train_ctx, self.config, distributed=self.distributed())
    
    def init_dataloaders(self):
        self.train_dataloader = AllamoDataLoader(
            config=self.config,
            rank=self.dp_rank,
            world_size=self.dp_world_size,
            train_split=True
        )

        if self.config.eval_iters > 0 and self.config.eval_interval > 0:
            self.val_dataloader = AllamoDataLoader(
                config=self.config,
                rank=self.dp_rank,
                world_size=self.dp_world_size,
                train_split=False
            )
        else:
            self.val_dataloader = None

    def init_training(self):
        attention_version.configure(self.config)
        self.model_spec = get_model_spec(self.config.model_type)
        self.checkpoint_manager = CheckpointManager(self.config, self.train_ctx, self.train_dataloader, self.model_spec)
        self.checkpoint_manager.init_checkpoint()
        self.train_dataloader.load_datasets()
        if self.val_dataloader:
            self.val_dataloader.load_datasets()
        self.model_config = create_model_config(self.config, self.model_spec)
        self.model_ctx = nullcontext()
        self.log_init_learning_rate()

        if self.config.dflash_config:
            self.draft_loss_scaling_factor = self.config.dflash_config.get("loss_scaling_factor", 0.1)
            self.draft_block_size = self.config.dflash_config["block_size"]
    
    def init_metrics_logger(self):
        self.metrics_logger = MetricsLogger(self.config, self.train_ctx)

    def freeze_model_params(self, model: BaseModel):
        model.freeze_model_params(
            self.config.freeze_embeddings,
            self.config.freeze_lm_head,
            self.config.freeze_layers,
            self.config.keep_layers_trainable
        )
        
    def log_init_learning_rate(self):
        if self.config.decay_lr:
            lr = calculate_learning_rate(self.train_ctx, self.config)
            logger.info(f"Cosing decay learning rate enabled. Currect learning rate: {lr}")
        else:
            logger.info(f"Using constant learning rate: {self.config.learning_rate}")
    
    def trigger_gc(self):
        gc.collect()
        torch.cuda.empty_cache()
    
    def should_evaluate(self):
        return self.config.eval_iters > 0 and self.config.eval_interval > 0 and self.train_ctx.iter_num % self.config.eval_interval == 0

    def should_save_last_checkpoint(self):
        return self.config.checkpoint_interval > 0 and self.train_ctx.iter_num > self.start_iter and self.train_ctx.iter_num % self.config.checkpoint_interval == 0
    
    def should_log_metrics(self):
        return self.config.log_interval > 0 and self.train_ctx.iter_num % self.config.log_interval == 0 and self.train_ctx.master_process
    
    def clip_grad_norm(self):
        return torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip).item()
    
    def has_next_iter_to_perform(self):
        if self.config.num_train_epochs is not None and self.train_dataloader.epoch >= self.config.num_train_epochs:
            return False
        return self.train_ctx.iter_num <= self.config.max_iters
    
    def calculate_eta(self):
        current_time = datetime.datetime.now()
        elapsed_time = current_time - self.start_timestamp
        elapsed_iters = self.train_ctx.iter_num - self.start_iter
        if elapsed_iters < 1:
            return 'N/A'
        avg_time_per_iter = elapsed_time.total_seconds() / elapsed_iters
        eta_seconds = math.ceil(avg_time_per_iter * (self.config.max_iters - self.train_ctx.iter_num))
        return format_seconds_as_time(eta_seconds)
        
    def run_checkpoint_hook_program(self, hook_program, current_epoch, ckpt_file_name): 
        env_variables = {
            "ALLAMO_EPOCH_HOOK_RUN_UUID": self.train_ctx.run_uuid,
            "ALLAMO_EPOCH_HOOK_TRAINING_UUID": self.train_ctx.training_uuid,
            "ALLAMO_EPOCH_HOOK_EPOCH": str(current_epoch),
            "ALLAMO_EPOCH_HOOK_ITERATION": str(self.train_ctx.iter_num),
            "ALLAMO_EPOCH_HOOK_MODEL_CKPT_PATH": str(os.path.abspath(get_model_checkpoint_path(ckpt_file_name, self.config.out_dir))),
            "ALLAMO_EPOCH_HOOK_CONFIG_CKPT_PATH": str(os.path.abspath(get_config_checkpoint_path(ckpt_file_name, self.config.out_dir)))
        }
        try:
            process = subprocess.Popen(hook_program, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env_variables)
            return process.pid
        except Exception as err:
            return f"n/a - Error: {err}"
    
    def dist_all_reduce(self, x: torch.Tensor, op: dist.ReduceOp):
        if self.distributed():
            dist.all_reduce(x, op=op)
        return x

    def compute_logits_and_loss(self, batch, last_gas_step):
        with self.model_ctx:
            return self.model(**batch)
    
    def model_forward_step(self, batch, last_gas_step):
        with self.model_ctx:
            return self.model(**batch)
    
    @torch.no_grad()
    def evaluate_val_loss(self):
        self.val_dataloader.reset_offset()
        self.model.eval()
        validation_metrics = torch.zeros(4).to(self.config.device)
        for _ in range(self.config.eval_iters):
            batch = self.val_dataloader.get_batch()
            model_output = self.forward_step(batch, 1, False)
            validation_metrics[0] += model_output.loss.item()
            validation_metrics[1] += model_output.unmasked_labels
            validation_metrics[2] += model_output.accuracy
            validation_metrics[3] += 1
        validation_metrics = self.dist_all_reduce(validation_metrics, op=dist.ReduceOp.SUM)
        assert int(validation_metrics[3].item()) == self.config.eval_iters * self.dp_world_size
        val_loss = validation_metrics[0] / validation_metrics[3]
        val_acc = validation_metrics[2] / validation_metrics[3]
        self.model.train()
        return val_loss, validation_metrics[1], val_acc
    
    def evaluate(self):
        eval_time = time.time()
        val_loss_t, unmasked_labels_t, val_acc_t = self.evaluate_val_loss()
        eval_time = time.time() - eval_time

        val_loss = val_loss_t.item()

        if self.train_ctx.iter_num > self.start_iter and val_loss < self.train_ctx.best_val_loss:
            self.train_ctx.best_val_loss = val_loss
            if self.config.save_best_checkpoint:
                self.save_checkpoint('ckpt')
                
        if self.train_ctx.master_process:
            val_acc = val_acc_t.item()
            unmasked_labels = int(unmasked_labels_t.item())
            val_ppl = torch.exp(val_loss_t).item()
            diff_best_losses = self.train_ctx.best_val_loss - self.train_ctx.best_train_loss
            logger.info(
                f"iter {self.train_ctx.iter_num:,}: val loss={val_loss:.4f} ppl={val_ppl:.4f} acc={val_acc:.4f} "
                f"(best loss={self.train_ctx.best_val_loss:.4f}, diff={diff_best_losses:.4f}), tokens {unmasked_labels:,}"
            )
            if self.config.log_metrics:
                self.metrics_logger.log_metrics({
                    "eval/time": eval_time*1000,
                    "eval/loss": val_loss,
                    "eval/ppl": val_ppl,
                    "eval/acc": val_acc,
                    "eval/tokens": unmasked_labels,
                    "eval/best_loss": self.train_ctx.best_val_loss,
                    "eval/diff_best_losses": diff_best_losses
                })
        self.trigger_gc()

    def supervised_forward_step(self, batch, gradient_accumulation_steps, last_gas_step):
        logits, draft_logits = self.model_forward_step(batch, last_gas_step)
        if "target_weights" not in batch or batch["target_weights"] is None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch["target_ids"].view(-1),
                ignore_index=self.config.ignore_index
            )
        else:
            assert draft_logits is None, "Draft logits are not supported for weighted loss"
            loss = (batch["target_weights"].view(-1) * F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch["target_ids"].view(-1),
                reduction="none"
            )).sum()
            if self.config.weighted_loss_method == 'openchat':
                target_weights = batch["target_weights"].sum()
                # sum loss weights over all processes
                target_weights = self.dist_all_reduce(target_weights, op=dist.ReduceOp.SUM)
                loss = (self.dp_world_size / target_weights) * loss
            else:
                loss = loss / torch.sum(batch["target_weights"] > 0).item()
        
        unmasked_labels = torch.sum(batch["target_ids"].view(-1) != self.config.ignore_index).item()
        accuracy = (logits.max(2).indices == batch["target_ids"]).sum().item() / unmasked_labels if unmasked_labels > 0 else 0

        draft_loss = None
        draft_accuracy = 0.0
        draft_unmasked_labels = 0
        draft_ignored_groups = 0
        draft_accepted_groups = 0
        if draft_logits is not None:
            B, T = batch["target_ids"].shape

            padded = F.pad(
                batch["target_ids"],
                (1, self.draft_block_size),
                value=self.config.ignore_index,
            )
            draft_labels = padded[:, 1:].unfold(1, self.draft_block_size, 1)[:, :T, :] # (B, T, draft_block_size)
            draft_labels[:, :, 0] = self.config.ignore_index

            # Mask groups containing EOS - tokens after EOS are unpredictable
            eos_token_id = self.config.dflash_config.get("eos_token_id")
            if eos_token_id is not None:
                group_has_eos = (draft_labels == eos_token_id).any(dim=-1, keepdim=True)
                draft_labels = draft_labels.masked_fill(group_has_eos, self.config.ignore_index)

            # If any label in a group is ignore_index, mask the entire group.
            group_has_ignore = (draft_labels == self.config.ignore_index).any(dim=-1, keepdim=True)
            draft_labels = draft_labels.masked_fill(group_has_ignore, self.config.ignore_index)
            draft_labels = draft_labels.contiguous().view(B, T * self.draft_block_size)

            draft_ignored_groups = group_has_ignore.squeeze(-1).sum().item()
            draft_total_groups = B * T
            draft_accepted_groups = draft_total_groups - draft_ignored_groups

            gamma = self.config.dflash_config.get("loss_decay_gamma", self.draft_block_size / math.log(10))
            k = torch.arange(self.draft_block_size, device=draft_logits.device)
            position_weights = torch.exp(-k / gamma)  # (draft_block_size,)

            weight_map = position_weights.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
            weight_map = weight_map.reshape(B, T * self.draft_block_size)
            weight_map = weight_map * (draft_labels != self.config.ignore_index).float()

            per_token_loss = F.cross_entropy(
                draft_logits.view(-1, draft_logits.size(-1)),
                draft_labels.view(-1),
                ignore_index=self.config.ignore_index,
                reduction="none",
            )
            draft_loss = (per_token_loss * weight_map.view(-1)).sum() / weight_map.sum().clamp(min=1)

            draft_unmasked_labels = torch.sum(draft_labels.view(-1) != self.config.ignore_index).item()
            draft_accuracy = (
                (draft_logits.view(-1, draft_logits.size(-1)).argmax(1) == draft_labels.view(-1)).sum().item()
                / draft_unmasked_labels
                if draft_unmasked_labels > 0 else 0.0
            )

        if gradient_accumulation_steps > 1:
            # scale the loss to account for micro steps
            loss = loss / gradient_accumulation_steps
            if draft_loss is not None:
                draft_loss = draft_loss / gradient_accumulation_steps
        
        return ModelOutput(
            loss=loss,
            accuracy=accuracy,
            unmasked_labels=unmasked_labels,
            draft_loss=draft_loss,
            draft_accuracy=draft_accuracy,
            draft_unmasked_labels=draft_unmasked_labels,
            draft_ignored_groups=draft_ignored_groups,
            draft_accepted_groups=draft_accepted_groups,
        )

    def dpo_forward_step(self, batch, gradient_accumulation_steps, last_gas_step):
        policy_chosen_logits, _ = self.model_forward_step({"input_ids": batch["chosen_input_ids"], "target_ids": batch["chosen_target_ids"]}, last_gas_step)
        policy_rejected_logits, _ = self.model_forward_step({"input_ids": batch["rejected_input_ids"], "target_ids": batch["rejected_target_ids"]}, last_gas_step)
        policy_chosen_logps = get_log_prob(policy_chosen_logits, batch["chosen_target_ids"], self.config.ignore_index)
        policy_rejected_logps = get_log_prob(policy_rejected_logits, batch["rejected_target_ids"], self.config.ignore_index)
        
        assert "reference_chosen_logps" in batch and batch["reference_chosen_logps"] is not None
        reference_chosen_logps = batch["reference_chosen_logps"]
        reference_rejected_logps = batch["reference_rejected_logps"]
        
        # calculate DPO loss
        chosen_rewards = self.config.dpo_chosen_beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.config.dpo_rejected_beta * (policy_rejected_logps - reference_rejected_logps)
        reward_penalty = self.config.dpo_penalty_lambda * torch.maximum(torch.zeros_like(policy_chosen_logps), reference_chosen_logps - policy_chosen_logps)
        dpo_loss = -F.logsigmoid(chosen_rewards - rejected_rewards - reward_penalty).mean()
        
        if gradient_accumulation_steps > 1:
            dpo_loss = dpo_loss / gradient_accumulation_steps # scale the loss to account for micro steps
            
        chosen_unmasked_labels = torch.sum(batch["chosen_target_ids"].view(-1) != self.config.ignore_index).item()
        rejected_unmasked_labels = torch.sum(batch["rejected_target_ids"].view(-1) != self.config.ignore_index).item()
        unmasked_labels = chosen_unmasked_labels + rejected_unmasked_labels
        
        accuracy = (policy_chosen_logits.max(2).indices == batch["chosen_target_ids"]).sum().item() / chosen_unmasked_labels if chosen_unmasked_labels > 0 else 0
        
        if last_gas_step and self.config.log_interval > 0 and self.train_ctx.iter_num % self.config.log_interval == 0:
            chosen_rewards = chosen_rewards.detach() 
            rejected_rewards = rejected_rewards.detach()
            reward_accuracies = (chosen_rewards > rejected_rewards).float().mean()
            reward_margins = (chosen_rewards - rejected_rewards).mean()
            chosen_rewards = chosen_rewards.mean()
            rejected_rewards = rejected_rewards.mean()
            reward_penalty = reward_penalty.mean()

            policy_chosen_logps = policy_chosen_logps.detach()
            policy_rejected_logps = policy_rejected_logps.detach()
            policy_accuracies = (policy_chosen_logps > policy_rejected_logps).float().mean()
            policy_chosen_logps = policy_chosen_logps.mean()
            policy_rejected_logps = policy_rejected_logps.mean()
            
            metrics = torch.tensor([
                1,
                reward_accuracies.item(),
                reward_margins.item(),
                chosen_rewards.item(),
                rejected_rewards.item(),
                reward_penalty.item(),
                policy_accuracies.item(),
                policy_chosen_logps.item(),
                policy_rejected_logps.item()
            ]).to(self.config.device)
            metrics = self.dist_all_reduce(metrics, op=dist.ReduceOp.SUM)
            
            if self.train_ctx.master_process:
                cnt = metrics[0].item()
                reward_accuracies = metrics[1].item() / cnt
                reward_margins = metrics[2].item() / cnt
                chosen_rewards = metrics[3].item() / cnt
                rejected_rewards = metrics[4].item() / cnt
                reward_penalty = metrics[5].item() / cnt
                policy_accuracies = metrics[6].item() / cnt
                policy_chosen_logps = metrics[7].item() / cnt
                policy_rejected_logps = metrics[8].item() / cnt
                if self.config.log_metrics:
                    self.metrics_logger.log_metrics({
                        "dpo/rewards/accuracies": reward_accuracies,
                        "dpo/rewards/margins": reward_margins,
                        "dpo/rewards/chosen": chosen_rewards,
                        "dpo/rewards/rejected": rejected_rewards,
                        "dpo/rewards/penalty": reward_penalty,
                        "dpo/logps/chosen": policy_chosen_logps,
                        "dpo/logps/rejected": policy_rejected_logps,
                        "dpo/logps/accuracies": policy_accuracies
                    })
                else:
                    logger.info(
                        f"iter {self.train_ctx.iter_num:,}: "
                        f"reward_acc={reward_accuracies:.4f} reward_marg={reward_margins:.4f} "
                        f"reward_chosen={chosen_rewards:.4f} reward_rejected={rejected_rewards:.4f} "
                        f"reward_penalty={reward_penalty:.4f}"
                    )
        
        return ModelOutput(
            loss=dpo_loss,
            accuracy=accuracy,
            unmasked_labels=unmasked_labels
        )
        
    def forward_step(self, batch, gradient_accumulation_steps, last_gas_step):
        if self.config.training_type == 'pre' or self.config.training_type == 'sft':
            return self.supervised_forward_step(batch, gradient_accumulation_steps, last_gas_step)
        elif self.config.training_type == 'dpo':
            return self.dpo_forward_step(batch, gradient_accumulation_steps, last_gas_step)
        else:
            raise ValueError(f"Unknown training type: {self.config.training_type}")
    
    def train(self):
        self.trigger_gc()
        logger.info(f"Training configuration:\n{pprint.pformat(self.config)}")
        total_batch_size = self.config.batch_size * self.config.gradient_accumulation_steps * self.dp_world_size
        total_tokens_per_iter = self.config.block_size * total_batch_size
        logger.info(f"Starting training (run id: {self.train_ctx.run_uuid}, world size: {self.train_ctx.world_size}, total batch size: {total_batch_size})")
        self.start_iter = self.train_ctx.iter_num
        self.start_timestamp = datetime.datetime.now()
        current_epoch = self.train_dataloader.epoch
        current_num_loaded_files = self.train_dataloader.get_num_loaded_files()
        iter_metrics = torch.zeros(9).to(self.config.device)
        batch = self.train_dataloader.get_batch() # fetch the very first batch
        while self.has_next_iter_to_perform():
            if current_epoch < self.train_dataloader.epoch:
                ckpt_file_name = f'epoch_{current_epoch}'
                self.save_checkpoint(ckpt_file_name, model_only=True, epoch_ckpt=True)
                if self.config.epoch_completion_hook_program and self.train_ctx.master_process:
                    pid = self.run_checkpoint_hook_program(self.config.epoch_completion_hook_program, current_epoch, ckpt_file_name)
                    logger.info(f"Epoch completion hook program started with pid {pid}")
                current_epoch = self.train_dataloader.epoch
                current_num_loaded_files = self.train_dataloader.get_num_loaded_files()
            elif self.config.save_checkpoint_on_dataset_reload and current_num_loaded_files != self.train_dataloader.get_num_loaded_files():
                ckpt_file_name = f'ds_reload_{current_epoch}-{current_num_loaded_files}'
                self.save_checkpoint(ckpt_file_name, model_only=True, epoch_ckpt=False)
                current_num_loaded_files = self.train_dataloader.get_num_loaded_files()
            elif self.config.should_override_config(self.train_ctx.iter_num):
                self.config.override_config_properties()
            

            # evaluate the loss on train/val sets and write best checkpoint
            if self.should_evaluate():
                self.evaluate()
                
            if self.should_save_last_checkpoint():
                ckpt_file_name = 'last_eval_ckpt'
                self.save_checkpoint(ckpt_file_name)
                if self.config.regular_checkpoint_hook_program and self.train_ctx.master_process:
                    pid = self.run_checkpoint_hook_program(self.config.regular_checkpoint_hook_program, current_epoch, ckpt_file_name)
                    logger.info(f"Regular checkpoint hook program started with pid {pid}")
            
            accuracy = 0
            iter_metrics.zero_()
            batch_mfu_excluded_time = 0
            timer = time.time()
            fwdbwd_time = time.time()
            # forward backward update, with optional gradient accumulation to simulate larger batch size
            for micro_step in range(self.config.gradient_accumulation_steps):
                model_output = self.forward_step(batch, self.config.gradient_accumulation_steps, (micro_step == self.config.gradient_accumulation_steps - 1))
                
                mfu_excluded_time = time.time()
                loss = model_output.loss
                iter_metrics[0] += model_output.loss.item()
                iter_metrics[1] += model_output.unmasked_labels
                iter_metrics[2] += model_output.accuracy
                iter_metrics[3] += 1

                if model_output.draft_loss is not None:
                    loss += model_output.draft_loss * self.draft_loss_scaling_factor
                    iter_metrics[5] += model_output.draft_loss.item()
                    iter_metrics[6] += model_output.draft_accuracy
                    iter_metrics[7] += model_output.draft_ignored_groups
                    iter_metrics[8] += model_output.draft_accepted_groups
                
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                batch = self.train_dataloader.get_batch()
                batch_mfu_excluded_time += time.time() - mfu_excluded_time
                
                # backward pass, with gradient scaling
                self.scaler.scale(loss).backward()
                
            # clip the gradient
            if self.config.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                iter_metrics[4] += self.clip_grad_norm()
            
            mfu_excluded_time = time.time()
            # sync loss and acc over all processes
            iter_metrics = self.dist_all_reduce(iter_metrics, op=dist.ReduceOp.SUM)
            
            # adjust learning rate
            lr = calculate_learning_rate(self.train_ctx, self.config)
            if self.config.adaptive_learning_rate:
                lr = lr * math.sqrt(iter_metrics[1].item() / total_tokens_per_iter)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            if self.train_ctx.master_process:
                # FIXME: CPU-GPU sync point, could be done when logging metrics
                self.train_ctx.processed_tokens += int(iter_metrics[1])
            batch_mfu_excluded_time += time.time() - mfu_excluded_time
            
            # step the optimizer and scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            self.optimizer.zero_grad(set_to_none=True)
            fwdbwd_time = time.time() - fwdbwd_time - batch_mfu_excluded_time

            if self.should_log_metrics():
                iter_time = time.time() - timer
                iter_metrics_cpu = iter_metrics.cpu()
                lossf = iter_metrics_cpu[0].item() / self.dp_world_size
                ppl = torch.exp(torch.tensor(lossf)).item()
                accuracy = iter_metrics_cpu[2].item() / iter_metrics_cpu[3].item()
                draft_lossf = iter_metrics_cpu[5].item() / self.dp_world_size
                draft_accuracy = iter_metrics_cpu[6].item() / iter_metrics_cpu[3].item()
                draft_ignored_groups = iter_metrics_cpu[7].item() / iter_metrics_cpu[3].item()
                draft_accepted_groups = iter_metrics_cpu[8].item() / iter_metrics_cpu[3].item()
                grad_norm = iter_metrics[4].item() / self.dp_world_size
                if self.config.mfu_flops_peak > 0 and self.train_ctx.iter_num > self.start_iter:
                    mfu = estimate_mfu(self.model_num_params, self.config, self.config.batch_size * self.config.gradient_accumulation_steps, fwdbwd_time)
                    mfu_str = f'{mfu*100:.2f}%'
                else:
                    mfu = -1.0
                    mfu_str = 'n/a'
                mtu = fwdbwd_time/iter_time # model time utilization
                iter_time_ms = iter_time * 1000
                logger.info(
                    f"iter {self.train_ctx.iter_num:,}: loss {lossf:.4f}, ppl {ppl:.4f}, acc {accuracy:.4f}, "
                    f"iter time {iter_time_ms:.2f}ms, tokens {self.train_ctx.processed_tokens:,}, lr {lr:.8f}, "
                    f"mfu {mfu_str}, mtu {mtu*100:.2f}%, epoch {self.train_dataloader.epoch}, "
                    f"ETA: {self.calculate_eta()}"
                )
                if lossf < self.train_ctx.best_train_loss:
                    self.train_ctx.best_train_loss = lossf

                if self.config.log_metrics:
                    metrics = {
                        "train/loss": lossf,
                        "train/acc": accuracy,
                        "train/ppl": ppl,
                        "train/grad_norm": grad_norm,
                        "train/lr": lr,
                        "train/mtu": mtu,
                        "train/tokens_per_sec": (total_tokens_per_iter/iter_time),
                        "train/tokens_per_gpu_per_sec": (total_tokens_per_iter/self.train_ctx.world_size/iter_time),
                        "train/tokens": self.train_ctx.processed_tokens,
                        "train/epoch": self.train_dataloader.epoch,
                        "train/best_loss": self.train_ctx.best_train_loss,
                        "train/ds_offset": self.train_dataloader.dataset_offset,
                        "train/iter_time": iter_time_ms,
                    }
                    if mfu > 0:
                        metrics['train/mfu'] = mfu
                    if draft_accepted_groups + draft_ignored_groups > 0:
                        metrics['train/draft_loss'] = draft_lossf
                        metrics['train/draft_acc'] = draft_accuracy
                        metrics['train/total_loss'] = lossf + draft_lossf * self.draft_loss_scaling_factor
                        metrics['train/draft_ignored_groups'] = draft_ignored_groups
                        metrics['train/draft_accepted_groups'] = draft_accepted_groups
                    self.metrics_logger.log_metrics(metrics)
            self.train_ctx.iter_num += 1
            
        training_time = format_seconds_as_time((datetime.datetime.now() - self.start_timestamp).total_seconds())
        logger.info(f"Training finished in {training_time}")
        
        ckpt_file_name = 'final_ckpt'
        self.save_checkpoint(ckpt_file_name, model_only=True, epoch_ckpt=True)
        if self.config.epoch_completion_hook_program and self.train_ctx.master_process:
            pid = self.run_checkpoint_hook_program(self.config.epoch_completion_hook_program, current_epoch, ckpt_file_name)
            logger.info(f"Epoch completion hook program started with pid {pid}")
