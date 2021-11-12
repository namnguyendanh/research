# -*- coding: utf-8
# Copyright (c) 2021 by Phuc Phan

import os
import torch
import logging
import subprocess as sp 

from denver.constants import *
from denver.utils.data_utils import save_json
from denver.utils.print_utils import print_boxed
from denver.hiperopt.defaults import hiperopt_config_default_registry

logger = logging.getLogger(__name__)

def str2bool(v):
    return str(v).lower() in ('yes', 'true', 't', '1')

def set_default_value(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = value

def set_default_values(dictionary, default_dictionary):
    # Set multiple default values

    for key, value in default_dictionary.items():
        set_default_value(dictionary, key, value)

def set_value(dictionary, key, value):
    dictionary[key] = value

def set_values(dictionary, parameters_dict):
    for key, _ in dictionary.items():
        if key in parameters_dict:
            set_value(dictionary, key, parameters_dict[key])

def update_hiperopt_params_with_defaults(model_type, hiperopt_params):
    default_config = get_hiperopt_config_default(model_type=model_type)
    set_default_values(hiperopt_params, default_config)

def get_from_registry(key, registry):
    if hasattr(key, 'lower'):
        key = key.lower()
    
    if key in registry:
        return registry[key]
    else:
        raise ValueError(f"Key `{key}` not supported, available options: {registry.keys()}")

def get_hiperopt_config_default(model_type):
    return get_from_registry(model_type, hiperopt_config_default_registry)
    
def get_available_gpus_cuda_string():
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        return ",".join(str(i) for i in range(num_gpus))
    return None

def get_available_gpu_memory():
    _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]

    COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
    try:
        memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
        memory_free_values = [int(x.split()[0])
                              for i, x in enumerate(memory_free_info)]
    except Exception as e:
        print('"nvidia-smi" is probably not installed.', e)

    return memory_free_values

def print_hiperopt_results(hiperopt_results):
    print_boxed(text='Hiperopt results')
    for hiperopt_result in hiperopt_results:
        print('score: {:.6f} | parameters: {}'.format(
            hiperopt_result['metric_score'][0], hiperopt_result['parameters']
        ))
    print("")

def save_hiperopt_stats(hiperopt_stats, hiperopt_dir):
    hiperopt_stats_fn = os.path.join(
        hiperopt_dir,
        'hiperopt_statistics.json'
    )
    save_json(hiperopt_stats_fn, hiperopt_stats)
