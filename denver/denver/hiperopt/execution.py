# -*- coding: utf-8
# Copyright (c) 2021 by Phuc Phan

import copy
import signal
import logging
import pandas as pd
import multiprocessing

from pathlib import Path
from typing import Union, List
from abc import ABC, abstractmethod

from denver.constants import *
from denver.api import DenverAPI
from denver.utils.print_utils import print_style_free
from denver.hiperopt.sampling import HiperotSampler, logger
from denver.hiperopt.utils import set_values, get_from_registry
from denver.hiperopt.utils import get_available_gpus_cuda_string, get_available_gpu_memory

logger = logging.getLogger(__name__)

class HiperoptExecutor(ABC):
    def __init__(self, hiperopt_sampler: HiperotSampler, metric: str) -> None:
        self.hiperopt_sampler = hiperopt_sampler
        self.metric = metric

    def sort_hiperopt_results(self, hiperopt_results):
        return sorted(
            hiperopt_results, key=lambda hp_res: hp_res['metric_score'], 
            reverse=self.hiperopt_sampler.goal == MAXIMIZE
        )
    
    def get_metric_score(self, eval_stats) -> float:
        return eval_stats[self.metric]

    @abstractmethod
    def execute(
        self, 
        config: Union[str, dict], 
        dataset: Union[str, Path, pd.DataFrame]=None, 
        skip_save_model: bool=False, 
        skip_save_log: bool=False, 
        seed: int=123, 
        gpu_memory_limit: int=None, 
        allow_parallel_threads: bool=True, 
        **kwargs
    ):
        pass

class SerialExecutor(HiperoptExecutor):
    def __init__(
        self, 
        hiperopt_sampler: HiperotSampler, 
        metric: str, 
        **kwargs
    ) -> None:
        HiperoptExecutor.__init__(self, hiperopt_sampler, metric)

    def execute(
        self, 
        config: Union[str, dict], 
        dataset: Union[str, Path, pd.DataFrame]=None,  
        skip_save_model: bool=False, 
        early_stopping_epochs: Union[bool, int]=False, 
        seed: int=123, 
        gpus: Union[str, int, List[int]]=None, 
        gpu_memory_limit: int=None, 
        allow_parallel_threads: bool=True, 
        **kwargs
    ):
        hiperopt_results = []
        trials = 0
        while not self.hiperopt_sampler.finished():
            sampled_parameters = self.hiperopt_sampler.sample_batch()

            metric_scores = []

            for i, parameters in enumerate(sampled_parameters):
                modified_config = substitute_parameters(copy.deepcopy(config), parameters)
                trial_id = trials + i
                print_style_free(message="Experiment: {} ".format(str(trial_id)))

                base_path = modified_config[TRAINING_PARAMS][BASE_PATH]
                modified_config[TRAINING_PARAMS][BASE_PATH] =  Path(base_path)/f'experiment_{trial_id}'
                eval_results = run_experiment(
                    modified_config, 
                    early_stopping_epochs=early_stopping_epochs, 
                    skip_save_model=skip_save_model, 
                )
                metric_score = self.get_metric_score(eval_results)
                metric_scores.append(metric_score)

                hiperopt_results.append({
                    'parameters': parameters, 
                    'metric_score': metric_score, 
                    'eval_results': eval_results
                })
            
            trials += len(sampled_parameters)

            self.hiperopt_sampler.update_batch(
                zip(sampled_parameters, metric_scores)
            )

        hiperopt_results = self.sort_hiperopt_results(hiperopt_results)

        return hiperopt_results


class ParallelExecutor(HiperoptExecutor):
    
    num_workers = 2
    epsilon = 0.01
    epsilon_memory = 100
    TF_REQUIRED_MEMORY_PER_WORKER = 100

    def __init__(
        self, 
        hiperopt_sampler: HiperotSampler, 
        metric: str, 
        num_workers: int=2, 
        epsilon: float=0.01, 
        **kwargs
    ) -> None:
        HiperoptExecutor.__init__(self, hiperopt_sampler, metric)
        
        self.num_workers = num_workers
        self.epsilon = epsilon
        self.queue = None
        self.queue_results = multiprocessing.Queue()

    @staticmethod
    def init_worker():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    def _run_experiment_gpu(self, hiperopt_dict, **kwargs):
        gpu_id_meta = self.queue.get()
        
        try:
            parameters = hiperopt_dict[PARAMETERS]
            hiperopt_dict['gpus'] = gpu_id_meta['gpu_id']
            hiperopt_dict["gpu_memory_limit"] = gpu_id_meta["gpu_memory_limit"]
            eval_results = run_experiment(**hiperopt_dict)
            metric_score = self.get_metric_score(eval_results)
        finally:
            self.queue.put(gpu_id_meta)

        return_dict = {
            'parameters': parameters, 
            'metric_score': metric_score, 
            'eval_results': eval_results
        }
        self.queue_results.put(return_dict)

        return return_dict

    def _run_experiment(self, hiperopt_dict, **kwargs):
        parameters = hiperopt_dict[PARAMETERS]
        eval_results = run_experiment(**hiperopt_dict)
        metric_score = self.get_metric_score(eval_results)

        return_dict = {
            'parameters': parameters, 
            'metric_score': metric_score, 
            'eval_results': eval_results
        }
        self.queue_results.put(return_dict)

        return return_dict

    def execute(
        self, 
        config: Union[str, dict], 
        dataset: Union[str, Path, pd.DataFrame]=None, 
        skip_save_model: bool=False, 
        early_stopping_epochs: Union[bool, int]=False, 
        seed: int=123, 
        gpus: Union[str, int, List[int]]=None, 
        gpu_memory_limit: int=None, 

        allow_parallel_threads: bool=True, 
        **kwargs
    ):
        ctx = multiprocessing.get_context('spawn')
        
        if gpus is None:
            gpus = get_available_gpus_cuda_string()

        if gpus is not None:
            num_available_cpus = ctx.cpu_count()

            if self.num_workers > num_available_cpus:
                logger.warning(
                    "num_workers={}, num_available_cpus={}. "
                    "To avoid bottlenecks setting num workers to be less "
                    "or equal to number of available cpus is suggested".format(
                        self.num_workers, num_available_cpus
                    )
                )

            if isinstance(gpus, int):
                gpus = str(gpus)
            elif isinstance(gpus, List):
                gpus = ','.join(str(i) for i in gpus)

            gpus = gpus.strip()
            gpu_ids = gpus.split(',')
            num_gpus = len(gpu_ids)

            available_gpu_memory_list = get_available_gpu_memory()
            gpu_ids_meta = {}

            if num_gpus < self.num_workers:
                fraction = (num_gpus / self.num_workers) - self.epsilon
                for gpu_id in gpu_ids:
                    available_gpu_memory = available_gpu_memory_list[int(gpu_id)]
                    required_gpu_memory = fraction * available_gpu_memory

                    if gpu_memory_limit is None:
                        logger.warning(
                            'Setting gpu_memory_limit to {} '
                            'as there available gpus are {} '
                            'and the num of workers is {} '
                            'and the available gpu memory for gpu_id '
                            '{} is {}'.format(
                                required_gpu_memory, num_gpus,
                                self.num_workers,
                                gpu_id, available_gpu_memory)
                        )
                        new_gpu_memory_limit = required_gpu_memory - \
                                               (
                                                       self.TF_REQUIRED_MEMORY_PER_WORKER * self.num_workers)
                    else:
                        new_gpu_memory_limit = gpu_memory_limit
                        if new_gpu_memory_limit > available_gpu_memory:
                            logger.warning(
                                'Setting gpu_memory_limit to available gpu '
                                'memory {} minus an epsilon as the value specified is greater than '
                                'available gpu memory.'.format(
                                    available_gpu_memory)
                            )
                            new_gpu_memory_limit = available_gpu_memory - self.epsilon_memory

                        if required_gpu_memory < new_gpu_memory_limit:
                            if required_gpu_memory > 0.5 * available_gpu_memory:
                                if available_gpu_memory != new_gpu_memory_limit:
                                    logger.warning(
                                        'Setting gpu_memory_limit to available gpu '
                                        'memory {} minus an epsilon as the gpus would be underutilized for '
                                        'the parallel processes otherwise'.format(
                                            available_gpu_memory)
                                    )
                                    new_gpu_memory_limit = available_gpu_memory - self.epsilon_memory
                            else:
                                logger.warning(
                                    'Setting gpu_memory_limit to {} '
                                    'as the available gpus are {} and the num of workers '
                                    'are {} and the available gpu memory for gpu_id '
                                    '{} is {}'.format(
                                        required_gpu_memory, num_gpus,
                                        self.num_workers,
                                        gpu_id, available_gpu_memory)
                                )
                                new_gpu_memory_limit = required_gpu_memory
                        else:
                            logger.warning(
                                'gpu_memory_limit could be increased to {} '
                                'as the available gpus are {} and the num of workers '
                                'are {} and the available gpu memory for gpu_id '
                                '{} is {}'.format(
                                    required_gpu_memory, num_gpus,
                                    self.num_workers,
                                    gpu_id, available_gpu_memory)
                            )

                    process_per_gpu = int(
                        available_gpu_memory / new_gpu_memory_limit)
                    gpu_ids_meta[gpu_id] = {
                        "gpu_memory_limit": new_gpu_memory_limit,
                        "process_per_gpu": process_per_gpu}
            else:
                for gpu_id in gpu_ids:
                    gpu_ids_meta[gpu_id] = {
                        "gpu_memory_limit": gpu_memory_limit,
                        "process_per_gpu": 1
                    }

            manager = ctx.Manager()
            self.queue = manager.Queue()

            for gpu_id in gpu_ids:
                process_per_gpu = gpu_ids_meta[gpu_id]["process_per_gpu"]
                gpu_memory_limit = gpu_ids_meta[gpu_id]["gpu_memory_limit"]
                for _ in range(process_per_gpu):
                    gpu_id_meta = {
                        "gpu_id": gpu_id,
                        "gpu_memory_limit": gpu_memory_limit
                    }
                    self.queue.put(gpu_id_meta)

        pool = ctx.Pool(self.num_workers, ParallelExecutor.init_worker)

        try:
            hiperopt_results = []
            trials = 0
            while not self.hiperopt_sampler.finished():
                sampled_parameters = self.hiperopt_sampler.sample_batch()

                hiperopt_parameters = []
                for i, parameters in enumerate(sampled_parameters):
                    modified_config = substitute_parameters(copy.deepcopy(config), parameters)

                    trial_id = trials + i

                    print_style_free(message="Experiment: {} ".format(str(trial_id)))

                    hiperopt_parameters.append(
                        dict(
                            parameters=parameters,
                            config=modified_config,
                            dataset=dataset,
                            skip_save_model=skip_save_model,
                            gpus=gpus,
                            gpu_memory_limit=gpu_memory_limit,
                            allow_parallel_threads=allow_parallel_threads,
                            seed=seed, 
                        )
                    )
                trials += len(sampled_parameters)

                batch_results = None

                try:
                    if gpus is not None:
                        batch_results = pool.map(self._run_experiment_gpu,
                                                hiperopt_parameters)
                    else:
                        batch_results = pool.map(self._run_experiment,
                                                hiperopt_parameters)
                except:
                    # jobs = []

                    # if gpus is not None:
                    #     for i in range(len(hiperopt_parameters)):
                    #         proc = ctx.Process(target=self._run_experiment_gpu, args=(hiperopt_parameters[i], ))
                    #         proc.start()
                    #         jobs.append(proc)
                    # else:
                    #     for i in range(len(hiperopt_parameters)):
                    #         proc = ctx.Process(target=self._run_experiment, args=(hiperopt_parameters[i], ))
                    #         proc.start()
                    #         jobs.append(proc)

                    # for proc in jobs:
                        # proc.join()
                    raise Warning(
                            f"Not supported running parallel. "
                            f"Please select the type of 'executor' is 'serial' to replace."
                        )

                self.hiperopt_sampler.update_batch(
                    (result["parameters"], result["metric_score"]) for result in batch_results)

                hiperopt_results.extend(batch_results)
        finally:
            pool.close()
            pool.join()

        hiperopt_results = self.sort_hiperopt_results(hiperopt_results)

        return hiperopt_results


def get_build_hiperopt_executor(executor_type):
    return get_from_registry(executor_type, executor_registry)

executor_registry = {
    'parallel': ParallelExecutor, 
    'serial': SerialExecutor
}

    
def substitute_parameters(config, parameters):
    set_values(
        config[TRAINING_PARAMS][HYPER_PARAMS], 
        parameters
    )
    set_values(
        config[MODEL][INPUT_FEATURES][ENCODER][ARGS], 
        parameters
    )

    return config

def run_experiment(
        config, 
        dataset=None, 
        pct: float=0.1, 
        skip_save_model: bool=False, 
        early_stopping_epochs: Union[bool, int]=False, 
        seed: int=123, 
        gpus: Union[str, int, List[int]]=None, 
        gpu_memory_limit: int = None,
        allow_parallel_threads: bool = True, 
        **kwargs
    ):
        eval_results = DenverAPI.experiment(
            config=config, 
            dataset=dataset, 
            skip_save_model=skip_save_model,
            early_stopping_epochs=early_stopping_epochs, 
            pct=pct, 
            seed=seed
        )

        return eval_results