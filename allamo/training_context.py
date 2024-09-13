import os
import uuid
from dataclasses import dataclass

@dataclass
class TrainingContext:
    
    dp: int = -1
    tp: int = 1
    pp: int = 1
    
    def __post_init__(self):
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        self.local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        self.run_uuid = str(uuid.uuid4())
        self.training_uuid = self.run_uuid
        self.iter_num = 0
        self.best_train_loss = 1e2
        self.best_val_loss = 1e2
        self.processed_tokens = 0
        
        self._validate()

    def _validate(self):
        if self.pp < 1:
            self.pp = 1
        if self.tp < 1:
            self.tp = 1
        if self.dp < 1:
            self.dp = self.world_size // (self.tp * self.pp)
        
        assert self.dp > 0
        assert self.tp == 1, f"tp({self.tp}) > 1 is not supported"
        assert self.pp == 1, f"pp({self.pp}) > 1 is not supported"
        assert self.dp * self.tp * self.pp == self.world_size, f"dp({self.dp}) * tp({self.tp}) * pp({self.pp}) != world_size({self.world_size})"

    @property
    def master_process(self):
        return self.rank == 0
