from allamo.configuration import AllamoConfiguration
from allamo.trainer.dpo_fsdp_trainer import DPOTrainer

if __name__ == '__main__':
    config = AllamoConfiguration()
    config.training_type = 'dpo'
    trainer = DPOTrainer(config)
    trainer.init_wandb()
    trainer.train()
    trainer.close()
