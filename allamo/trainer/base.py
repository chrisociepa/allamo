import gc
import os
import time
import math
import datetime
import subprocess

import torch
import torch.distributed as dist
import torch.nn.functional as F

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
        self.data_loader = AllamoDataLoader(self.config, self.dp_rank, self.dp_world_size)
        
        self.init_training()
        self.init_metrics_logger()

    def distributed(self):
        raise NotImplementedError("Not implemented")

    def init_torch(self):
        self.device_type = 'cuda' if 'cuda' in self.config.device else 'cpu'
        init_torch(self.train_ctx, self.config, distributed=self.distributed())

    def init_training(self):
        attention_version.configure(self.config)
        self.model_spec = get_model_spec(self.config.model_type)
        self.checkpoint_manager = CheckpointManager(self.config, self.train_ctx, self.data_loader, self.model_spec)
        self.checkpoint_manager.init_checkpoint()
        self.data_loader.load_datasets()
        self.model_config = create_model_config(self.config, self.model_spec)
    
    def init_metrics_logger(self):
        self.metrics_logger = MetricsLogger(self.config, self.train_ctx)

    def freeze_model_params(self, model: BaseModel):
        model.freeze_model_params(
            self.config.freeze_embeddings,
            self.config.freeze_lm_head,
            self.config.freeze_layers,
            self.config.keep_layers_trainable
        )
            
    def init_gradient_accumulation_scheduler(self):
        if self.config.grad_accum_schedule: 
            self.config.grad_accum_max = self.config.gradient_accumulation_steps
            self.config.gradient_accumulation_steps = self.config.grad_accum_initial
            logger.info(
                f"Gradient accumulation scheduler enabled. "
                f"Current gradient accumulation steps: {self.config.gradient_accumulation_steps}"
            )
        self.gradient_accumulation_steps = self.config.gradient_accumulation_steps
        
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
        return self.config.eval_interval > 0 and self.train_ctx.iter_num % self.config.eval_interval == 0 and self.config.training_type != "dpo"

    def should_save_last_checkpoint(self):
        return self.config.checkpoint_interval > 0 and self.train_ctx.iter_num > self.start_iter and self.train_ctx.iter_num % self.config.checkpoint_interval == 0
    
    def should_log_metrics(self):
        return self.config.log_interval > 0 and self.train_ctx.iter_num % self.config.log_interval == 0 and self.train_ctx.master_process
    
    def clip_grad_norm(self):
        return torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip).item()
    
    def has_next_iter_to_perform(self):
        if self.config.num_train_epochs is not None and self.data_loader.epoch >= self.config.num_train_epochs:
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
    
    def get_grad_accum(self):
        if self.config.grad_accum_schedule and self.gradient_accumulation_steps < self.config.grad_accum_max and self.train_ctx.iter_num % (self.config.grad_accum_max_iter/100) == 0:
            return min(self.gradient_accumulation_steps + 1, self.config.grad_accum_max)
        else:
            return self.gradient_accumulation_steps
        
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

    def compute_logits_and_loss(self, batch, last_micro_step):
        return self.model(**batch)
    
    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss(self):
        losses_out = {}
        accuraces = {}
        self.model.eval()
        for split in self.data_loader.splits:
            validation_metrics = torch.zeros(3).to(self.config.device)
            for _ in range(self.config.eval_iters):
                batch = self.data_loader.get_batch(split, True)
                logits, loss, _ = self.compute_logits_and_loss(batch, False)
                if batch["target_weights"] is not None:
                    loss = loss / torch.sum(batch["target_weights"] > 0).item()
                validation_metrics[0] += loss.item()
                validation_metrics[1] += (logits.max(2).indices == batch["target_ids"]).sum().item() / torch.sum(batch["target_ids"].view(-1) != self.config.ignore_index).item()
                validation_metrics[2] += 1
            validation_metrics = self.dist_all_reduce(validation_metrics, op=dist.ReduceOp.SUM)
            losses_out[split] = validation_metrics[0] / validation_metrics[2]
            accuraces[split] = validation_metrics[1] / validation_metrics[2]
        self.model.train()
        if 'val' not in losses_out:
            losses_out['val'] = losses_out['train']
            accuraces['val'] = accuraces['train']
        return losses_out, accuraces
    
    def evaluate(self):
        eval_time = time.time()
        losses, accuraces = self.estimate_loss()
        eval_time = time.time() - eval_time
        train_loss = losses['train'].item()
        val_loss = losses['val'].item()
        if self.train_ctx.iter_num > self.start_iter:
            if train_loss < self.train_ctx.best_train_loss:
                self.train_ctx.best_train_loss = train_loss
            if val_loss < self.train_ctx.best_val_loss:
                self.train_ctx.best_val_loss = val_loss
                if self.config.save_best_checkpoint:
                    self.save_checkpoint('ckpt')
                
        if self.train_ctx.master_process:                
            train_ppl = torch.exp(losses['train']).item()
            val_ppl = torch.exp(losses['val']).item()
            logger.info(
                f"iter {self.train_ctx.iter_num:,}: train loss={train_loss:.4f} ppl={train_ppl:.4f} "
                f"acc={accuraces['train']:.4f} (best loss={self.train_ctx.best_train_loss:.4f}), "
                f"val loss={val_loss:.4f} ppl={val_ppl:.4f} acc={accuraces['val']:.4f} "
                f"(best loss={self.train_ctx.best_val_loss:.4f}), tokens {self.train_ctx.processed_tokens:,}"
            )
            if self.config.log_metrics:
                self.metrics_logger.log_metrics({
                    "eval/time": eval_time*1000,
                    "eval/samples_per_second": (self.config.eval_iters * len(self.data_loader.splits)) / eval_time,
                    "eval/train_loss": train_loss,
                    "eval/val_loss": val_loss,
                    "eval/train_ppl": train_ppl,
                    "eval/val_ppl": val_ppl,
                    "eval/train_acc": accuraces['train'].item(),
                    "eval/val_acc": accuraces['val'].item(),
                    "eval/diff_loss": (val_loss-train_loss),
                    "eval/diff_acc": (accuraces['train']-accuraces['val']).item(),
                    "eval/diff_ppl": (val_ppl-train_ppl),
                    "eval/best_train_loss": self.train_ctx.best_train_loss,
                    "eval/best_val_loss": self.train_ctx.best_val_loss
                })
        self.trigger_gc()

    def evaluate_pretraining_loss(self, batch, last_micro_step):
        logits, loss, _ = self.compute_logits_and_loss(batch, last_micro_step)
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps # scale the loss to account for micro steps
        
        unmasked_labels = self.config.block_size # we assume no padding in pretraining
        accuracy = (logits.max(2).indices == batch["target_ids"]).sum().item() / self.config.block_size
        return loss, unmasked_labels, accuracy

    def evaluate_sft_loss(self, batch, last_micro_step):
        logits, loss, _ = self.compute_logits_and_loss(batch, last_micro_step)
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps # scale the loss to account for micro steps
        
        if batch["target_weights"] is not None:
            if self.config.weighted_loss_method == 'openchat':
                target_weights = batch["target_weights"].sum()
                # sum loss weights over all processes
                target_weights = self.dist_all_reduce(target_weights, op=dist.ReduceOp.SUM)
                loss = (self.dp_world_size / target_weights) * loss
            else:
                loss = loss / torch.sum(batch["target_weights"] > 0).item()
        
        unmasked_labels = torch.sum(batch["target_ids"].view(-1) != self.config.ignore_index).item()
        accuracy = (logits.max(2).indices == batch["target_ids"]).sum().item() / unmasked_labels
        return loss, unmasked_labels, accuracy

    def evaluate_dpo_loss(self, batch, last_micro_step):
        policy_chosen_logits, _, _ = self.compute_logits_and_loss({"input_ids": batch["chosen_input_ids"], "target_ids": batch["chosen_target_ids"]}, last_micro_step)
        policy_rejected_logits, _, _ = self.compute_logits_and_loss({"input_ids": batch["rejected_input_ids"], "target_ids": batch["rejected_target_ids"]}, last_micro_step)
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
        
        if self.gradient_accumulation_steps > 1:
            dpo_loss = dpo_loss / self.gradient_accumulation_steps # scale the loss to account for micro steps
            
        chosen_unmasked_labels = torch.sum(batch["chosen_target_ids"].view(-1) != self.config.ignore_index).item()
        rejected_unmasked_labels = torch.sum(batch["rejected_target_ids"].view(-1) != self.config.ignore_index).item()
        unmasked_labels = chosen_unmasked_labels + rejected_unmasked_labels
        
        accuracy = (policy_chosen_logits.max(2).indices == batch["chosen_target_ids"]).sum().item() / chosen_unmasked_labels
        
        if last_micro_step and self.config.log_interval > 0 and self.train_ctx.iter_num % self.config.log_interval == 0:
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
        return dpo_loss, unmasked_labels, accuracy
        
    def evaluate_loss(self, batch, last_micro_step):
        if self.config.training_type == 'pre':
            return self.evaluate_pretraining_loss(batch, last_micro_step)
        elif self.config.training_type == 'sft':
            return self.evaluate_sft_loss(batch, last_micro_step)
        elif self.config.training_type == 'dpo':
            return self.evaluate_dpo_loss(batch, last_micro_step)
        else:
            raise ValueError(f"Unknown training type: {self.config.training_type}")
    
    def train(self):
        logger.info(f"Starting training (run id: {self.train_ctx.run_uuid}, world size: {self.train_ctx.world_size}) with configuration:\n{self.config}")
        batch = self.data_loader.get_batch('train') # fetch the very first batch
        self.start_iter = self.train_ctx.iter_num
        self.start_timestamp = datetime.datetime.now()
        current_epoch = self.data_loader.epoch
        current_num_loaded_files = self.data_loader.get_num_loaded_files()
        iter_metrics = torch.zeros(5).to(self.config.device)
        self.trigger_gc()
        while self.has_next_iter_to_perform():
            if current_epoch < self.data_loader.epoch:
                ckpt_file_name = f'epoch_{current_epoch}'
                self.save_checkpoint(ckpt_file_name, model_only=True, epoch_ckpt=True)
                if self.config.epoch_completion_hook_program and self.train_ctx.master_process:
                    pid = self.run_checkpoint_hook_program(self.config.epoch_completion_hook_program, current_epoch, ckpt_file_name)
                    logger.info(f"Epoch completion hook program started with pid {pid}")
                current_epoch = self.data_loader.epoch
                current_num_loaded_files = self.data_loader.get_num_loaded_files()
            elif self.config.save_checkpoint_on_dataset_reload and current_num_loaded_files != self.data_loader.get_num_loaded_files():
                ckpt_file_name = f'ds_reload_{current_epoch}-{current_num_loaded_files}'
                self.save_checkpoint(ckpt_file_name, model_only=True, epoch_ckpt=False)
                current_num_loaded_files = self.data_loader.get_num_loaded_files()
            elif self.config.should_override_config(self.train_ctx.iter_num):
                self.config.override_config_properties()
            
            timer = time.time()
                            
            # determine and set batch_size and gradient_accumulation_steps for this iteration 
            micro_batch_size = self.data_loader.update_batch_size(self.train_ctx.iter_num)
            total_batch_size = self.config.block_size * micro_batch_size * self.gradient_accumulation_steps * self.dp_world_size
            self.gradient_accumulation_steps = self.get_grad_accum()

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
            fwdbwd_time = time.time()
            # forward backward update, with optional gradient accumulation to simulate larger batch size
            for micro_step in range(self.gradient_accumulation_steps):
                loss, unmasked_labels, accuracy = self.evaluate_loss(batch, (micro_step == self.gradient_accumulation_steps - 1))
                
                mfu_excluded_time = time.time()
                iter_metrics[0] += loss.item()
                iter_metrics[1] += unmasked_labels
                iter_metrics[2] += accuracy
                iter_metrics[3] += 1
                
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                batch = self.data_loader.get_batch('train')
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
                lr = lr * math.sqrt(iter_metrics[1].item() / total_batch_size)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            if self.train_ctx.master_process:
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
                # get loss as float. note: this is a CPU-GPU sync point
                lossf = iter_metrics[0].item() / self.dp_world_size
                ppl = torch.exp(torch.tensor(lossf)).item()
                accuracy = iter_metrics[2].item() / iter_metrics[3].item()
                grad_norm = iter_metrics[4].item() / self.dp_world_size
                if self.config.mfu_flops_peak > 0 and self.train_ctx.iter_num > self.start_iter:
                    mfu = estimate_mfu(self.model_num_params, self.config, micro_batch_size * self.gradient_accumulation_steps, fwdbwd_time)
                    mfu_str = f'{mfu*100:.2f}%'
                else:
                    mfu = -1.0
                    mfu_str = 'n/a'
                mtu = fwdbwd_time/iter_time # model time utilization
                iter_time_ms = iter_time * 1000
                logger.info(
                    f"iter {self.train_ctx.iter_num:,}: loss {lossf:.4f}, ppl {ppl:.4f}, acc {accuracy:.4f}, "
                    f"iter time {iter_time_ms:.2f}ms, tokens {self.train_ctx.processed_tokens:,}, lr {lr:.8f}, "
                    f"mfu {mfu_str}, mtu {mtu*100:.2f}%, epoch {self.data_loader.epoch}, "
                    f"ETA: {self.calculate_eta()}"
                )

                if self.config.log_metrics:
                    metrics = {
                        "train/loss": lossf,
                        "train/acc": accuracy,
                        "train/ppl": ppl,
                        "train/grad_norm": grad_norm,
                        "train/lr": lr,
                        "train/mtu": mtu,
                        "train/tokens_per_sec": (total_batch_size/iter_time),
                        "train/tokens_per_gpu_per_sec": (total_batch_size/self.train_ctx.world_size/iter_time),
                        "train/tokens": self.train_ctx.processed_tokens,
                        "train/epoch": self.data_loader.epoch,
                        "train/total_batch_size": total_batch_size,
                        "train/iter_time": iter_time_ms,
                    }
                    if mfu > 0:
                        metrics['train/mfu'] = mfu
                    if self.config.dataset_seq_train:
                        metrics['train/ds_offset'] = self.data_loader.dataset_offset
                    self.metrics_logger.log_metrics(metrics)
            self.train_ctx.iter_num += 1
            
        training_time = format_seconds_as_time((datetime.datetime.now() - self.start_timestamp).total_seconds())
        logger.info(f"Training finished in {training_time}")
        
        ckpt_file_name = 'final_ckpt'
        self.save_checkpoint(ckpt_file_name, model_only=True, epoch_ckpt=True)
        if self.config.epoch_completion_hook_program and self.train_ctx.master_process:
            pid = self.run_checkpoint_hook_program(self.config.epoch_completion_hook_program, current_epoch, ckpt_file_name)
            logger.info(f"Epoch completion hook program started with pid {pid}")
