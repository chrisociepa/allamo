from allamo.configuration import AllamoConfiguration
from allamo.logging import configure_logger, logger
from allamo.trainer.simple_trainer import SimpleTrainer
from allamo.trainer.fsdp_trainer import FSDPTrainer
from allamo.training_context import TrainingContext

if __name__ == '__main__':
    config = AllamoConfiguration()
    train_ctx = TrainingContext(
        tp = config.tensor_parallel_degree,
    )
    if train_ctx.master_process:
        configure_logger(config, True)

    if train_ctx.world_size > 1 and config.fsdp_sharding_strategy != 'None':
        trainer = FSDPTrainer(config, train_ctx)
    else:
        trainer = SimpleTrainer(config, train_ctx)
    
    trainer.train()
    trainer.close()
