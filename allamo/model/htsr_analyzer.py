from allamo.configuration import AllamoConfiguration
from allamo.model.model import AllamoTransformer
from allamo.logging import logger

class HTSRAnalyzer:
    
    def __init__(self, config: AllamoConfiguration, model, optimizer):
        self.report = None
        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.watcher = None
        if config.htsr_analysis and self.model is not None and self.optimizer is not None:
            import weightwatcher as ww
            self.watcher = ww.WeightWatcher()
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
            logger.info("HTSR analysis completed: overfit=%d, underfit=%d, stable=%d",
                        overfit_groups, underfit_groups, stable_groups)
    
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
