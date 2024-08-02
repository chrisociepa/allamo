"""
This single file is intended to perform some magic for DPO training using FSDP.
"""
import datetime
import gc
import os
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from model import AllamoTransformer
from configuration import AllamoConfiguration

from copy import deepcopy
from fsdp_train import AllamoFSDPTrainer
from train_utils import model_checkpoint_files_exist

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
            
            self.logger.info("Configuring reference model with FSDP")
            ref_model = FSDP(ref_model, **self.fsdp_config)
            self.logger.info(f"Reference model configured with FSDP and sharding strategy {self.sharding_strategy}")
            
            # compile the model - requires PyTorch 2.0
            if config.compile:
                self.logger.info("compiling the reference model... (takes a ~minute)")
                try:
                    ref_model = torch.compile(ref_model, mode=config.compile_mode)
                    self.logger.info("Reference model compiled and ready to use")
                except Exception as err:
                    self.logger.warning(f"Reference model cannot be compiled, because torch.compile is not supported: {err}")
            self.ref_model = ref_model
            self.ref_model.eval()
        else:
            self.ref_model = None
            self.logger.warning("Reference model checkpoint not provided. Reference log probabilities must be supplied via DataLoader")
        
    def calculate_preference_loss(self,
        policy_chosen_logps,
        policy_rejected_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        beta=0.5
    ):
        prefered_relative_logps = policy_chosen_logps - reference_chosen_logps
        disprefered_relative_logps = policy_rejected_logps - reference_rejected_logps
        
        reward_accuracies = (prefered_relative_logps > disprefered_relative_logps).float().mean(dim=-1)
        reward_margins = (prefered_relative_logps - disprefered_relative_logps).mean(dim=-1)
        
        loss = -F.logsigmoid(beta * (prefered_relative_logps - disprefered_relative_logps)).mean(dim=-1)
        
        return loss, prefered_relative_logps.mean(dim=-1), disprefered_relative_logps.mean(dim=-1), reward_accuracies, reward_margins
    
    def forward(self, batch, last_micro_step):
        timer = time.time()
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
                reference_chosen_logits, _, _ = self.model(input_ids=batch["chosen_input_ids"], target_ids=batch["chosen_target_ids"])
                reference_rejected_logits, _, _ = self.model(input_ids=batch["rejected_input_ids"], target_ids=batch["rejected_target_ids"])
                reference_chosen_logps = get_log_prob(reference_chosen_logits, batch["chosen_target_ids"], self.config.ignore_index)
                reference_rejected_logps = get_log_prob(reference_rejected_logits, batch["rejected_target_ids"], self.config.ignore_index)
        
        # calculate DPO loss
        chosen_rewards = self.config.preference_beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.config.preference_beta * (policy_rejected_logps - reference_rejected_logps)
        
        dpo_loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
        
        if self.gradient_accumulation_steps > 1:
            dpo_loss = dpo_loss / self.gradient_accumulation_steps # scale the loss to account for micro steps
            
        chosen_unmasked_labels = torch.sum(batch["chosen_target_ids"].view(-1) != self.config.ignore_index).item()
        rejected_unmasked_labels = torch.sum(batch["rejected_target_ids"].view(-1) != self.config.ignore_index).item()
        unmasked_labels = chosen_unmasked_labels + rejected_unmasked_labels
        
        chosen_accuracy = (policy_chosen_logits.max(2).indices == batch["chosen_target_ids"]).sum().item()
        rejected_accuracy = (policy_rejected_logits.max(2).indices == batch["rejected_target_ids"]).sum().item()
        accuracy = (chosen_accuracy + rejected_accuracy) / unmasked_labels
        
        if last_micro_step and self.config.log_interval > 0 and self.iter_num % self.config.log_interval == 0:
            chosen_rewards = chosen_rewards.detach() 
            rejected_rewards = rejected_rewards.detach()
            reward_accuracies = (chosen_rewards > rejected_rewards).float().mean()
            reward_margins = (chosen_rewards - rejected_rewards).mean()
            chosen_rewards = chosen_rewards.mean()
            rejected_rewards = rejected_rewards.mean()
            
            metrics = torch.tensor([
                reward_accuracies.item(),
                reward_margins.item(),
                chosen_rewards.item(),
                rejected_rewards.item()
            ]).to(self.config.device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            
            if self.master_process:
                fwd_time = time.time() - timer
                if self.config.wandb_log:
                    wandb.log({
                        "iter": self.iter_num,
                        "dpo/forward_time": fwd_time*1000,
                        "dpo/loss": dpo_loss.item(),
                        "dpo/rewards/accuracies": reward_accuracies.item(),
                        "dpo/rewards/margins": reward_margins.item(),
                        "dpo/rewards/chosen": chosen_rewards.item(),
                        "dpo/rewards/rejected": rejected_rewards.item()
                    })
                else:
                    self.logger.info(
                        f"iter {self.iter_num:,}: dpo loss={dpo_loss:.4f} "
                        f"reward_acc={reward_accuracies:.4f} reward_marg={reward_margins:.4f} "
                        f"reward_chosen={chosen_rewards:.4f} reward_rejected={rejected_rewards:.4f} "
                    )
        return dpo_loss, unmasked_labels, accuracy

if __name__ == '__main__':
    config = AllamoConfiguration()
    config.training_type = 'dpo'
    trainer = DPOAllamoFSDPTrainer(config)
    
    # logging
    if config.wandb_log and trainer.master_process:
        import wandb
        wandb_run_name = config.wandb_run_name + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        wandb.init(project=config.wandb_project, name=wandb_run_name, config=config)
    
    # clean up after initialization
    gc.collect()
    torch.cuda.empty_cache()
    
    trainer.train()  
    
    dist.barrier()
    dist.destroy_process_group()
