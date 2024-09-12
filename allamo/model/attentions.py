import torch
from allamo.configuration import AllamoConfiguration
from allamo.logging import logger

class AttentionVersion:
    """
    Versions:
    0 - eager
    1 - SDPA
    2 - FA2
    """
    
    def __init__(self):
        self.disable_flash_attn_2()
        
    def configure(self, config: AllamoConfiguration):
        if config.attention_implementation:
            if config.attention_implementation == 'flash_attention_2':
                self.enable_flash_attn_2()
            elif config.attention_implementation == 'sdpa':
                self.disable_flash_attn_2()
            elif config.attention_implementation == 'eager':
                self.force_eager()
    
    def disable_flash_attn_2(self):
        self.version = 1 if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else 0
        self.flash_attn_2_supports_window_size = False
    
    def enable_flash_attn_2(self):
        self.version = 2
        self.flash_attn_2_supports_window_size = True
        
    def force_eager(self):
        self.version = 0
        self.flash_attn_2_supports_window_size = False
    
    def log_version(self, sliding_window):
        if self.version == 2:
            logger.info("Using Flash Attention 2")
            if self.flash_attn_2_supports_window_size and sliding_window:
                logger.info("Using sliding window")
        elif self.version == 1:
            logger.info("Using scaled_dot_product_attention")
        elif self.version == 0:
            logger.info("WARNING: using slow attention")
        else:
            raise Exception('Unsupported Flash Attention version!')
    
attention_version = AttentionVersion()
