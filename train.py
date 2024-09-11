from allamo.configuration import AllamoConfiguration
from allamo.trainer.simple_trainer import SimpleTrainer

if __name__ == '__main__':
    config = AllamoConfiguration()
    trainer = SimpleTrainer(config)
    trainer.init_wandb()
    trainer.train()
    trainer.close()
