import inspect
import torch
from allamo.configuration import AllamoConfiguration
from allamo.logging import logger

class AttentionVersion:
    """
    Versions:
    0 - eager
    1 - SDPA
    2 - FA2
    3 - FA3
    4 - xformers
    5 - FA2_custom_mask
    """
    
    def __init__(self):
        self.attn_impl_module = None
        self.enable_sdpa()
        
    def configure(self, config: AllamoConfiguration):
        if config.attention_implementation:
            if config.attention_implementation == 'sdpa':
                self.enable_sdpa()
            elif config.attention_implementation == 'fa2':
                self.enable_flash_attn_2()
            elif config.attention_implementation == 'fa2_custom_mask':
                self.enable_flash_attn_2_custom_mask()
                if config.dropout > 0:
                    logger.warning("Flash Attention 2 with custom masks does not support dropout")
            elif config.attention_implementation == 'fa3':
                self.enable_flash_attn_3()
                if config.dropout > 0:
                    logger.warning("Flash Attention 3 does not support dropout")
            elif config.attention_implementation == 'xformers':
                self.enable_xformers()
            elif config.attention_implementation == 'eager':
                self.force_eager()
    
    def enable_sdpa(self):
        self.version = 1 if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else 0
        self.flash_attn_supports_window_size = False
    
    def enable_flash_attn_2(self):
        self.version = 2
        try:            
            import flash_attn
            self.flash_attn_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
            self.attn_impl_module = flash_attn
        except ImportError:
            self.enable_sdpa()
            logger.warning("Flash Attention 2 is not available, falling back to scaled_dot_product_attention!")
        
    def enable_flash_attn_3(self):
        self.version = 3
        try:
            import flash_attn_interface
            self.flash_attn_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
            self.attn_impl_module = flash_attn_interface
        except ImportError:
            self.enable_sdpa()
            logger.warning("Flash Attention 3 is not available, falling back to scaled_dot_product_attention!")
        
    def enable_xformers(self):
        self.version = 4
        try:
            import xformers.ops as xops
            self.attn_impl_module = xops
        except ImportError:
            self.enable_sdpa()
            logger.warning("xformers is not available, falling back to scaled_dot_product_attention!")
        self.flash_attn_supports_window_size = False # TODO: check xops.fmha.attn_bias.LowerTriangularFromBottomRightLocalAttentionMask

    def enable_flash_attn_2_custom_mask(self):
        self.version = 5
        try:
            import fa2_custom_mask
            self.attn_impl_module = fa2_custom_mask
        except ImportError:
            self.enable_sdpa()
            logger.warning("Flash Attention 2 with custom masks is not available, falling back to scaled_dot_product_attention!")
        self.flash_attn_supports_window_size = False
        
    def force_eager(self):
        self.version = 0
        self.flash_attn_supports_window_size = False
    
    def log_version(self, sliding_window):
        if self.version == 2:
            logger.info("Using Flash Attention 2")
            if self.flash_attn_supports_window_size and sliding_window:
                logger.info("Using sliding window")
        elif self.version == 3:
            logger.info("Using Flash Attention 3")
            if self.flash_attn_supports_window_size and sliding_window:
                logger.info("Using sliding window")
        elif self.version == 1:
            logger.info("Using scaled_dot_product_attention")
        elif self.version == 4:
            logger.info("Using xformers memory_efficient_attention")
        elif self.version == 5:
            logger.info("Using Flash Attention 2 with custom masks")
            if sliding_window:
                logger.warning("Sliding window is not supported with Flash Attention 2 with custom masks")
        elif self.version == 0:
            logger.info("WARNING: using slow attention")
        else:
            raise Exception('Unsupported Flash Attention version!')
    
attention_version = AttentionVersion()
