from allamo.configuration import AllamoConfiguration
from allamo.trainer.fsdp_trainer import FSDPTrainer

if __name__ == '__main__':
    config = AllamoConfiguration()
    trainer = FSDPTrainer(config)
    trainer.train()
    trainer.close()
