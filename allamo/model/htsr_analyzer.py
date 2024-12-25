import wandb
from allamo.configuration import AllamoConfiguration
from allamo.model.model import AllamoTransformer
from allamo.logging import logger
from allamo.training_context import TrainingContext

class HTSRAnalyzer:
    
    def __init__(self, config: AllamoConfiguration, train_ctx: TrainingContext, model, optimizer):
        self.report = None
        self.config = config
        self.train_ctx = train_ctx
        self.model = model
        self.optimizer = optimizer
        self.watcher = None
        if config.htsr_analysis and self.model is not None and self.optimizer is not None:
            from weightwatcher import WeightWatcher
            self.watcher = WeightWatcher()
            logger.info("HTSR analysis enabled")
        
    def analyze(self):
        if self.watcher is None:
            return
        logger.info("run HTSR analysis")
        details = self.watcher.analyze(model=self.model)
        if details.empty:
            self.report = None
            logger.warning("No result for HTSR analysis")
        elif not {'longname', 'alpha'}.issubset(details.columns):
            self.report = None
            logger.warning("HTSR analysis canceled. Invalid result format - expected 'longname' and 'alpha' columns but got: " + ",".join(details.columns))
        else:
            log_spectral_norm_avg = details['log_spectral_norm'].mean().item()
            log_norm_avg = details['log_norm'].mean().item()
            log_alpha_norm_avg = details['log_alpha_norm'].mean().item()
            stable_rank_avg = details['stable_rank'].mean().item()

            self.report = details.set_index('longname')['alpha'].to_dict()
            overfit_groups = 0
            underfit_groups = 0
            stable_groups = 0
            for alpha in self.report.values():
                if alpha < 2.0:
                    overfit_groups += 1
                elif alpha > 6.0:
                    underfit_groups += 1
                else:
                    stable_groups += 1
            
            if self.train_ctx.master_process:
                total_alpha_sum = sum(alpha for alpha in self.report.values())
                total_alpha_avg = total_alpha_sum / len(self.report)
                overfit_alpha_sum = sum(alpha for alpha in self.report.values() if alpha < 2.0)
                overfit_alpha_avg = overfit_alpha_sum / len(self.report)
                underfit_alpha_sum = sum(alpha for alpha in self.report.values() if alpha > 6.0)
                underfit_alpha_avg = underfit_alpha_sum / len(self.report)
                stable_alpha_sum = sum(alpha for alpha in self.report.values() if alpha >= 2.0 and alpha <= 6.0)
                stable_alpha_avg = stable_alpha_sum / len(self.report)
                logger.info("Groups: overfit=%d, underfit=%d, stable=%d. Alphas: total=%.2f, overfit=%.2f, underfit=%.2f, stable=%.2f. Log_spectral_norm=%.2f, log_alpha_norm=%.2f, stable_rank=%.2f",
                            overfit_groups, underfit_groups, stable_groups, total_alpha_avg, overfit_alpha_avg, underfit_alpha_avg, stable_alpha_avg, log_spectral_norm_avg, log_alpha_norm_avg, stable_rank_avg)
                
                if self.config.wandb_log:
                    wandb.log({
                        "iter": self.train_ctx.iter_num,
                        "htsra/groups/overfit": overfit_groups,
                        "htsra/groups/underfit": underfit_groups,
                        "htsra/groups/stable": stable_groups,
                        "htsra/alphas/total": total_alpha_avg,
                        "htsra/alphas/overfit": overfit_alpha_avg,
                        "htsra/alphas/underfit": underfit_alpha_avg,
                        "htsra/alphas/stable": stable_alpha_avg,
                        "htsra/log_spectral_norm": log_spectral_norm_avg,
                        "htsra/log_norm_norm": log_norm_avg,
                        "htsra/log_alpha_norm": log_alpha_norm_avg,
                        "htsra/stable_rank": stable_rank_avg,
                    })
    
    def adjust_learning_rate(self):
        if self.report:
            overfit_groups = 0
            underfit_groups = 0
            stable_groups = 0
            ignored_groups = 0
            for param_group in self.optimizer.param_groups:
                n = param_group["name"]
                if n in self.report:
                    alpha = self.report[n]
                    if alpha < 2.0:
                        overfit_groups += 1
                        param_group["lr"] *= self.config.htsr_smin
                    elif alpha > 6.0:
                        underfit_groups += 1
                        param_group["lr"] *= self.config.htsr_smax
                    else:
                        stable_groups += 1
                else:
                    ignored_groups += 1
            logger.debug("Adjusted LR for groups based on HTSR analysis: overfit=%d, underfit=%d, stable=%d, ignored=%d",
                        overfit_groups, underfit_groups, stable_groups, ignored_groups)
