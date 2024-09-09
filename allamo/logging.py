import datetime
import logging
import os
from allamo.configuration import AllamoConfiguration

logger = logging.getLogger()

def configure_logger(config: AllamoConfiguration = None, with_file_handler: bool = True):
    log_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(log_formatter)
    logger.addHandler(stream_handler)
    
    if with_file_handler:
        run_timestamp_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        log_file_path = os.path.join(config.out_dir, f'allamo-{run_timestamp_str}.log')
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(log_formatter)
        logger.addHandler(file_handler)
