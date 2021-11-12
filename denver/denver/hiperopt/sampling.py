# -*- coding: utf-8
# Copyright (c) 2021 by Phuc Phan

import copy
import logging
import numpy as np

from abc import ABC, abstractmethod
from bayesmark.space import JointSpace
from typing import List, Dict, Tuple, Iterable, Any
from bayesmark.builtin_opt.pysot_optimizer import PySOTOptimizer

from denver.constants import *
from denver.hiperopt.utils import str2bool, get_from_registry

logger = logging.getLogger(__name__)

class HiperotSampler(ABC):
    def __init__(self, goal: str, parameters: Dict[str, Any]) -> None:
        assert goal in [MINIMIZE, MAXIMIZE]
        self.goal = goal
        self.parameters = parameters

    @abstractmethod
    def sample(self) -> Dict[str, Any]:
        # Yields a set of parameters names and their values.
        # Define `build_hyperopt_strategy` which would take paramters as inputs
        pass

    def sample_batch(self, batch_size: int=1) -> Dict[str, Any]:
        samples = []
        for _ in range(batch_size):
            try:
                samples.append(self.sample())
            except IndexError:
                # Logic: is samples is empty it means that we encountered
                # the IndexError the first time we called self.sample()
                # so we should raise the exception. If samples is not empty
                # we should just return it, even if it will contain
                # less samples than the specified batch_size.
                if not samples:
                    raise IndexError
                
        return samples

    @abstractmethod
    def update(self, sampled_parameters: Dict[str, Any], metric_score: float):
        # Given the results of previous computation, it updates the strategy by Bayesian
        pass

    def update_batch(self, parameters_metric_tuples: Iterable[Tuple[Dict[str, Any], float]]):
        for (sampled_parameters, metric_score) in parameters_metric_tuples:
            self.update(sampled_parameters, metric_score)

    @abstractmethod
    def finished(self) -> bool:
        # Should return true when all samples have been sampled
        pass

class RandomSampler(HiperotSampler):
    def __init__(
        self, 
        goal: str, 
        parameters: Dict[str, Any], 
        num_samples=10, 
        **kwargs
    ) -> None:
        HiperotSampler.__init__(self, goal, parameters)

        params_for_join_space = copy.deepcopy(parameters)

        cat_params_values_types = {}
        
        for param_name, param_values in params_for_join_space.items():
            if param_values[TYPE] == CATEGORY:
                param_values[TYPE] = 'cat'
                values_str = []
                values_types = {}
                for value in param_values['values']:
                    value_str = str(value)
                    values_str.append(value_str)
                    value_type = type(value)
                    if value_type == bool:
                        value_type = str2bool
                    values_types[value_str] = value_type

                param_values['values'] = values_str
                cat_params_values_types[param_name] = param_values

            if param_values[TYPE] == FLOAT:
                param_values[TYPE] = 'real'

            if param_values[TYPE] == INT or param_values[TYPE] == REAL:
                if SPACE not in param_values:
                    param_values[SPACE] = 'linear'
                if isinstance(param_values['range'], List):
                    param_values['range'] = (param_values['range'][0], param_values['range'][1])
        
        self.cat_params_values_types = cat_params_values_types
        self.space = JointSpace(params_for_join_space)
        self.num_samples = num_samples
        self.samples = self._determine_samples()
        self.COUNT = 0

    def _determine_samples(self):
        samples = []
        for _ in range(self.num_samples):
            bnds = self.space.get_bounds()
            x = bnds[:, 0] + (bnds[:, 1] - bnds[:, 0]) * np.random.rand(1, len(
                self.space.get_bounds()))

            sample = self.space.unwarp(x)[0]
            samples.append(sample)

        return samples

    def sample(self) -> Dict[str, Any]:
        if self.COUNT >= len(self.samples):
            raise IndexError()

        sample = self.samples[self.COUNT]
        for key in sample:
            if key in self.cat_params_values_types:
                values_types = self.cat_params_values_types[key]
                sample[key] = values_types[sample[key]](sample[key])

        self.COUNT += 1
        return sample

    def update(self, sampled_parameters: Dict[str, Any], metric_score: float):
        pass

    def finished(self) -> bool:
        return self.COUNT >= len(self.samples)

class PySOTSampler(HiperotSampler):
    """
    This is a wrapper around the pySOT package (https://github.com/dme65/pySOT)
    """
    def __init__(
        self, 
        goal: str, 
        parameters: Dict[str, Any], 
        num_samples: int=10, 
        **kwargs
    ) -> None:
        HiperotSampler.__init__(self, goal, parameters)

        params_for_join_space = copy.deepcopy(parameters)

        cat_params_values_types = {}
        
        for param_name, param_values in params_for_join_space.items():
            if param_values[TYPE] == CATEGORY:
                param_values[TYPE] = 'cat'
                values_str = []
                values_types = {}
                for value in param_values['values']:
                    value_str = str(value)
                    values_str.append(value_str)
                    value_type = type(value)
                    if value_type == bool:
                        value_type = str2bool
                    values_types[value_str] = value_type

                param_values['values'] = values_str
                cat_params_values_types[param_name] = param_values

            if param_values[TYPE] == FLOAT:
                param_values[TYPE] = 'real'

            if param_values[TYPE] == INT or param_values[TYPE] == REAL:
                if SPACE not in param_values:
                    param_values[SPACE] = 'linear'
                if isinstance(param_values['range'], List):
                    param_values['range'] = (param_values['range'][0], param_values['range'][1])
        
        self.cat_params_values_types = cat_params_values_types
        self.pysot_optimizer = PySOTOptimizer(params_for_join_space)
        self.COUNT = 0
        self.num_samples = num_samples

    def sample(self) -> Dict[str, Any]:
        if self.COUNT >= self.num_samples:
            raise IndexError()

        sample = self.pysot_optimizer.suggest(n_suggestions=1)[0]
        for key in sample:
            if key in self.cat_params_values_types:
                values_types = self.cat_params_values_types[key]
                sample[key] = values_types[sample[key]](sample[key])

        self.COUNT += 1
        return sample

    def update(self, sampled_parameters: Dict[str, Any], metric_score: float):
        for key in sampled_parameters:
            if key in self.cat_params_values_types:
                sampled_parameters[key] = str(sampled_parameters[key])

        self.pysot_optimizer.observe([sampled_parameters], metric_score)

    def finished(self) -> bool:
        return self.COUNT  >= self.num_samples

def get_build_hiperopt_sampler(strategy_type):
    return get_from_registry(strategy_type, sampler_registry)

sampler_registry = {
    'random': RandomSampler, 
    'pysot': PySOTSampler, 
}