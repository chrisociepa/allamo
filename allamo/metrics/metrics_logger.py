import datetime
import os
import time
import hashlib
import torch.distributed as dist
from typing import Dict
from allamo.configuration import AllamoConfiguration
from allamo.logging import logger
from allamo.training_context import TrainingContext

class MetricsLogger:

    def __init__(self, config: AllamoConfiguration, train_ctx: TrainingContext):
        self.config = config
        self.train_ctx = train_ctx
        self.run = None
        self.system_metrics_monitor = None

        if self.config.log_metrics:
            run_name = self.config.metrics_logger_run_name + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            if self.config.metrics_logger == 'wandb' and self.train_ctx.master_process:
                import wandb
                self.run = wandb.init(project=self.config.metrics_logger_project, name=run_name, config=self.config)
            
            if self.config.metrics_logger == 'neptune':
                from neptune_scale import NeptuneLoggingHandler, Run

                if self.train_ctx.world_size > 1:
                    # To correctly monitor each GPU usage
                    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(self.train_ctx.rank)

                    if self.config.metrics_logger_run_id:
                        custom_run_id = self.config.metrics_logger_run_id
                    else:
                        # Ensure that all processes log metadata to the same run
                        if self.train_ctx.rank == 0:
                            custom_run_id = [self.generate_run_id(run_name)]
                        else:
                            custom_run_id = [None]

                        dist.broadcast_object_list(custom_run_id, src=0)
                        custom_run_id = custom_run_id[0]

                    self.run = Run(
                        experiment_name=run_name,
                        project=self.config.metrics_logger_project,
                        monitoring_namespace=f"monitoring/training/rank_{self.train_ctx.rank}",
                        run_id=custom_run_id,
                    )
                else:
                    custom_run_id = self.config.metrics_logger_run_id if self.config.metrics_logger_run_id else self.generate_run_id(run_name)
                    self.run = Run(
                        experiment_name=run_name,
                        project=self.config.metrics_logger_project,
                        monitoring_namespace=f"monitoring/training/rank_{self.train_ctx.rank}",
                        run_id=custom_run_id,
                    )

                if self.train_ctx.master_process:
                    self.run.log_configs(self.config)
                    
                    npt_handler = NeptuneLoggingHandler(run=self.run)
                    logger.addHandler(npt_handler)
                
                if self.config.metrics_logger_hardware_monitoring:
                    from allamo.metrics.neptune_hardware_monitoring import SystemMetricsMonitor
                    self.system_metrics_monitor = SystemMetricsMonitor(self.run, namespace="rank_" + str(self.train_ctx.rank))
                    self.system_metrics_monitor.start()
    
    def generate_run_id(self, run_name: str):
        return hashlib.md5((run_name + str(time.time())).encode()).hexdigest()
    
    def log_metrics(self, metrics: Dict):
        if self.run is not None:
            if self.config.metrics_logger == 'wandb':
                metrics['iter'] = self.train_ctx.iter_num
                self.run.log(metrics)
            
            if self.config.metrics_logger == 'neptune':
                self.run.log_metrics(data=metrics, step=self.train_ctx.iter_num)
    
    def close(self):
        if self.run is not None:
            if self.config.metrics_logger == 'wandb':
                self.run.finish()
            
            if self.config.metrics_logger == 'neptune':
                self.run.close()

            if self.system_metrics_monitor is not None:
                self.system_metrics_monitor.stop()
