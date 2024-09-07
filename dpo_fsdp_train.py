"""
This single file is intended to perform some magic for DPO training using FSDP.
"""
import datetime
import gc
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from copy import deepcopy
from allamo.logging import logger
from allamo.model import AllamoTransformer
from allamo.configuration import AllamoConfiguration
from allamo.train_utils import model_checkpoint_files_exist
from fsdp_train import AllamoFSDPTrainer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def get_log_prob(logits, target_ids, ignore_index):
    """
    Args:
        logits: unnormalized logits [B, T, V]
        target_ids: masked labels [B, T]
        ignore_index: masked label id
    Returns:
        aggregated log probabilities [B, ]
    """
    labels = target_ids.clone()
    loss_mask = (labels != ignore_index)
    labels[labels == ignore_index] = 0 # will be ignored for the loss calc
    
    log_probs = F.log_softmax(logits, dim=-1)
    per_token_logps = torch.gather(log_probs, -1, labels.unsqueeze(-1)).squeeze(-1)
    return (per_token_logps * loss_mask).sum(-1)

class DPOAllamoFSDPTrainer(AllamoFSDPTrainer):

    def __init__(self, config: AllamoConfiguration):
        super().__init__(config)
        
    def __init_training(self, config: AllamoConfiguration):
        super().__init_training(config)
        if model_checkpoint_files_exist(config.reference_checkpoint_name, self.checkpoint_dir):
            ref_model_conf = deepcopy(self.model.config)
            ref_model = AllamoTransformer(ref_model_conf)
            self.load_model_checkpoint(ref_model, os.path.join(self.checkpoint_dir, f'model_{config.reference_checkpoint_name}.pt'), config)
            
            logger.info("Configuring reference model with FSDP")
            ref_model = FSDP(ref_model, **self.fsdp_config)
            logger.info(f"Reference model configured with FSDP and sharding strategy {self.sharding_strategy}")
            
            # compile the model - requires PyTorch 2.0
            if config.compile:
                logger.info("compiling the reference model... (takes a ~minute)")
                try:
                    ref_model = torch.compile(ref_model, mode=config.compile_mode)
                    logger.info("Reference model compiled and ready to use")
                except Exception as err:
                    logger.warning(f"Reference model cannot be compiled, because torch.compile is not supported: {err}")
            self.ref_model = ref_model
            self.ref_model.eval()
        else:
            self.ref_model = None
            logger.warning("Reference model checkpoint not provided. Reference log probabilities must be supplied via DataLoader")
    
    def forward(self, batch, last_micro_step):
        policy_chosen_logits, _, _ = self.model(input_ids=batch["chosen_input_ids"], target_ids=batch["chosen_target_ids"])
        policy_rejected_logits, _, _ = self.model(input_ids=batch["rejected_input_ids"], target_ids=batch["rejected_target_ids"])
        policy_chosen_logps = get_log_prob(policy_chosen_logits, batch["chosen_target_ids"], self.config.ignore_index)
        policy_rejected_logps = get_log_prob(policy_rejected_logits, batch["rejected_target_ids"], self.config.ignore_index)
        
        if "reference_chosen_logps" in batch and batch["reference_chosen_logps"] is not None:
            reference_chosen_logps = batch["reference_chosen_logps"]
            reference_rejected_logps = batch["reference_rejected_logps"]
        else:
            assert self.ref_model is not None
            with torch.no_grad():
                reference_chosen_logits, _, _ = self.ref_model(input_ids=batch["chosen_input_ids"], target_ids=batch["chosen_target_ids"])
                reference_rejected_logits, _, _ = self.ref_model(input_ids=batch["rejected_input_ids"], target_ids=batch["rejected_target_ids"])
                reference_chosen_logps = get_log_prob(reference_chosen_logits, batch["chosen_target_ids"], self.config.ignore_index)
                reference_rejected_logps = get_log_prob(reference_rejected_logits, batch["rejected_target_ids"], self.config.ignore_index)
        
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
        
        if last_micro_step and self.config.log_interval > 0 and self.iter_num % self.config.log_interval == 0:
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
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            
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
                if self.config.wandb_log:
                    wandb.log({
                        "iter": self.iter_num,
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
                        f"iter {self.iter_num:,}: "
                        f"reward_acc={reward_accuracies:.4f} reward_marg={reward_margins:.4f} "
                        f"reward_chosen={chosen_rewards:.4f} reward_rejected={rejected_rewards:.4f} "
                        f"reward_penalty={reward_penalty:.4f}"
                    )
        return dpo_loss, unmasked_labels, accuracy

if __name__ == '__main__':
    config = AllamoConfiguration()
    config.training_type = 'dpo'
    trainer = DPOAllamoFSDPTrainer(config)
    
    # logging
    if config.wandb_log and trainer.train_ctx.master_process:
        wandb_run_name = config.wandb_run_name + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        wandb.init(project=config.wandb_project, name=wandb_run_name, config=config)
    
    # clean up after initialization
    gc.collect()
    torch.cuda.empty_cache()
    
    trainer.train()
    trainer.close()
