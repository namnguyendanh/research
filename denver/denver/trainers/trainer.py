# -*- coding: utf-8
# Copyright (c) 2021 by Phuc Phan

import os
import logging
import datetime

from typing import Union
from pathlib import Path

from denver.utils.print_utils import *
from denver.learners import DenverLearner

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

__types__ = ['class', 'ner']

class ModelTrainer:
    '''A class for Model trainer. '''

    def __init__(
        self, 
        learn: DenverLearner=None, 
        use_tensorboard: bool = False,
    ):
        """Initialize a Model Trainer

        :param learn: the learn of the model that you want to train. The learn should inherit from DenverLearner
        :param use_tensorboard: If True, writes out tensorboard information
        """
        self.learn = learn
        self.use_tensorboard = use_tensorboard

    def train(
        self,
        base_path: Union[str, Path]=None,
        is_save_best_model: bool=True,
        model_file: str=None, 
        learning_rate: float=2e-2,
        batch_size: float=128,
        num_epochs: int=15,
        monitor_test: bool = False,
        verbose: bool=True,
        skip_save_model: bool=False, 
        overwrite: bool = False,
        **kwargs
    ):
        """
        Trains any learners.

        :param base_path: The path to model directory
        :param is_save_best_model: If True save the best model, else save the final model
        :param model_file: The file name to save the model
        :param learning_rate: Initial learning rate
        :param batch_size: Size of batches during training
        :param num_epochs: Number of epochs to train
        :param monitor_test: If True, test data is evaluated at end of each epoch
        :param verbose: If False, disable show report results
        :param skip_save_model: If True, disables saving the model
        :param overwrite: If True, overwrite the old file
        """
        try:
            print_line(text='training')
            now = datetime.datetime.now()

            if not os.path.exists(base_path):
                logger.debug(f"Folder `{base_path}` not exists. Make folder!")
                os.makedirs(base_path)

            self.learn.train(base_path=base_path,
                             is_save_best_model=is_save_best_model,
                             model_file=model_file,
                             learning_rate=learning_rate, 
                             batch_size=batch_size, 
                             num_epochs=num_epochs, 
                             skip_save_model=skip_save_model, 
                             **kwargs)
            
            print_style_time(message="The trained time: {}".format(datetime.datetime.now() - now))

            results = self.learn.validate()
            if results and verbose:
                print_style_free(message="Evaluated Valid: ")
                view_table(results)

            if monitor_test and self.learn.data_source.test is not None:
                results = self.learn.evaluate(self.learn.data_source.test.data)
                if results and verbose:
                    print_style_free(message="Evaluated Test: ")
                    view_table(results)
                return results
                    
            return None
        except KeyboardInterrupt:
            logger.info(f"-"*100)
            logger.info(f"Exiting from training early.")
