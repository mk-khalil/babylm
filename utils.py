import random
import numpy as np
import torch
from transformers import set_seed

import logging
from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import os
import datetime

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.
    Args:
        seed (`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class Logger:
    def __init__(self, log_file_name, tb_log_dir=None):
        """
        Logger with ability to log to console, file and tensorboard.
        """
        # Make sure log_file_name exists since logging throws error if it does not exist
        if log_file_name is not None:
            os.makedirs(os.path.dirname(log_file_name), exist_ok=True)

        # Set global logging level (i.e. which log levels will be ignored)
        logging.basicConfig(level=logging.DEBUG)
        # logging.basicConfig(filename=log_file_name, filemode='w', format='%(name)s - %(levelname)s - %(message)s')

        # Create a custom logger
        self.file_logger = logging.getLogger('MyFileLogger')
        self.console_logger = logging.getLogger('MyConsoleLogger')

        # Create handlers
        c_handler = logging.StreamHandler() # writes logs to stdout (console)
        f_handler = logging.FileHandler(log_file_name, mode='w') # writes logs to a file
        c_handler.setLevel(logging.DEBUG)
        f_handler.setLevel(logging.DEBUG)

        # Create formatters and add it to handlers
        c_format = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the corresponding loggers
        self.console_logger.addHandler(c_handler)
        self.file_logger.addHandler(f_handler)

        # Initialize TensorBoard --------------------------------------------------------------------
        if tb_log_dir is not None:
            # log_file_dir = '/'.join(log_file_name.split('/')[:-1])
            run_name =  datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.tb_summaryWriter = SummaryWriter(tb_log_dir + run_name)

            self.tb_global_steps = defaultdict(lambda : 0)

    def log_msg_to_console(self, msg):
        """
        Passed in message will be logged to console.
        """
        mgs = '\n' + msg + '\n'
        self.console_logger.info(msg)

    def log_dict_to_file(self, info_dict):
        """
        key and value pairs in the info_dict will be logged into the file.
        """
        log_str = '\n'
        log_str += '='*20 + '\n'
        for k,v in info_dict.items():
            log_str += f'{k}: {v} \n'
        log_str += '='*20
        log_str += '\n'
        self.file_logger.info(log_str)

    def log_to_file(self, entity):
        log_str = '\n'
        log_str += '='*20 + '\n'
        log_str += str(entity) +'\n'
        log_str += '='*20
        log_str += '\n'
        self.file_logger.info(log_str)
    
    def log_scalar_to_tb(self, tag, scalar_value):
        # tb_summaryWriter.add_scalar("Training Loss", cur_epoch_loss, i)
        global_step = self.tb_global_steps[tag] # get current global_step for the tag
        self.tb_summaryWriter.add_scalar(tag, scalar_value, global_step) # log scalar to tb
        self.tb_global_steps[tag] = self.tb_global_steps[tag] + 1 # update global_step for the tag