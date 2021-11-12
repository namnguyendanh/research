# -*- coding: utf-8
# Copyright (c) 2021 by Phuc Phan

import os
import yaml
import logging


from pathlib import Path
from pprint import pformat
from typing import Union, List

from denver.constants import *
from denver import DENVER_VERSION
from denver.utils.print_utils import print_boxed
from denver.hiperopt.sampling import get_build_hiperopt_sampler
from denver.hiperopt.execution import get_build_hiperopt_executor
from denver.hiperopt.utils import update_hiperopt_params_with_defaults
from denver.hiperopt.utils import print_hiperopt_results, save_hiperopt_stats

logger = logging.getLogger(__name__)

def hiperopt(
    config: Union[str, dict, Path], 
    dataset: Union[str, Path]=None,  
    skip_save_model: bool=True, 
    skip_save_hiperopt_statistics: bool=False, 
    seed: int=123, 
    gpus: Union[str, int, List[int]]=None, 
    gpu_memory_limit: int = None,
    allow_parallel_threads: bool = True, 
    **kwargs
):
    """This method performs an hyperparameter optimization.

    :param base_path: Path to the output dictionary
    :param config: Config which defines the different parameters of the model, features, 
                   preprocessing and training.  If `str`, filepath to yaml configuration file.   
    :param dataset: If not None, Path to dataset to be used in the experiment.
                    else use dataset from config.
    :param skip_save_model: Disables saving model weights and hyperparameters each time 
                            the model experiments.
    :param skip_save_hyperopt_statistics: Skips saving hyperopt stats file.
    :param seed: Random seed used for weights initialization, splits and any other 
                 random function (int: default: 42).
    :param gpu_memory_limit: Maximum memory in MB to allocate per GPU-device (int, default:None).
    :param allow_parallel_threads: Allow to use multithreading parallelism to 
                                improve performance at the cost of determinism.
    
    :return: The results for the hyperparameter optimization (List[dict]).

    """
    
    # check if config is a path or a dict
    if isinstance(config, str):
        with open(config, 'r') as f:
            config = yaml.safe_load(f)

    if HIPEROPT not in config:
        hiperopt_config = {}
        logger.warning(
            f"Hiperopt section not present in config! Setup default hiperopt config."
        )

    else:
        hiperopt_config = config[HIPEROPT]

    # get name type of the model
    model_type = config[MODEL][INPUT_FEATURES][ENCODER][NAME]
    # update hiperopt params with default values
    update_hiperopt_params_with_defaults(model_type, hiperopt_config)

    # print hiperopt config
    logger.debug(
        f"- Hiperopt config: {pformat(config)}"
    )
    if hiperopt_config:
        sampler = hiperopt_config[SAMPLER]
        executor = hiperopt_config[EXECUTOR]
        parameters = hiperopt_config[PARAMETERS]
        metric = hiperopt_config[METRIC]
        goal = hiperopt_config[GOAL]
        early_stopping_epochs = hiperopt_config[RUN].get('early_stopping_epochs', False)
        skip_save_model = hiperopt_config[RUN].get('skip_save_model', skip_save_model)
        gpus = hiperopt_config[RUN].get('gpus', gpus)
    else:
        raise ValueError(f"'hiperopt_config' is a None value. "
                        f"You must defined in `config` file or use defaults config is provied.")

    hiperopt_sampler = get_build_hiperopt_sampler(sampler[TYPE])(goal, parameters, **sampler)

    hiperopt_executor = get_build_hiperopt_executor(executor[TYPE])(hiperopt_sampler, metric, **executor)

    # print_denver(message='Hello !', denver_version=DENVER_VERSION)
    print_boxed(text='RUNNING HIPEROPT')

    hiperopt_results = hiperopt_executor.execute(
        config, 
        dataset=dataset, 
        skip_save_model=skip_save_model, 
        early_stopping_epochs=early_stopping_epochs, 
        gpus=gpus, 
        gpu_memory_limit=gpu_memory_limit, 
        allow_parallel_threads=allow_parallel_threads, 
        seed=seed
    )

    print_hiperopt_results(hiperopt_results)
    
    output_directory = Path(config[TRAINING_PARAMS][BASE_PATH])/'hiperopt_stats/'

    if not skip_save_hiperopt_statistics:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        hiperopt_stats = {
            'hiperopt_config': hiperopt_config, 
            'hiperopt_results': hiperopt_results
        }

        save_hiperopt_stats(hiperopt_stats, output_directory)
        logger.info(
            'Hiperopt stats saved to: {}'.format(output_directory)
        )

    logger.info("Finished hiperopt. ")

    return hiperopt_results
