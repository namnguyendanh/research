# -*- coding: utf-8
# Copyright (c) 2021 by Phuc Phan

# Denver toolbox

import os
import sys
import torch
import logging
import warnings
import matplotlib

from fastai.text import *

matplotlib.use('template', warn=False)

__authors__ = 'phucpx'
__all__ = []

warnings.filterwarnings("ignore")

# disable logging of allennlp
logging.getLogger('allennlp.training.util').disabled = True
logging.getLogger('allennlp.common.params').disabled = True
logging.getLogger('allennlp.common.checks').disabled = True
logging.getLogger('allennlp.data.vocabulary').disabled = True
logging.getLogger('allennlp.nn.initializers').disabled = True
logging.getLogger('allennlp.training.trainer').disabled = True
logging.getLogger('allennlp.common.from_params').disabled = True
logging.getLogger('allennlp.training.optimizers').disabled = True
logging.getLogger('allennlp.training.checkpointer').disabled = True
logging.getLogger('allennlp.training.tensorboard_writer').disabled = True
logging.getLogger('allennlp.data.iterators.data_iterator').disabled = True
logging.getLogger('allennlp.modules.token_embedders.embedding').disabled = True

logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout)

formatter = logging.Formatter(
    "{asctime} {levelname}  {name}:{lineno} - {message}", style="{"
)
stream_handler.setFormatter(formatter)

logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

DENVER_VERSION = '₍ᵥ₃.₀.₂₎'
# print_denver(message='Hello ! ', denver_version=DENVER_VERSION)

HOME = os.path.expanduser('~')
DENVER_DIR  = os.path.join(HOME, '.denver')

IC_DIR       = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT    = os.path.dirname(IC_DIR)

MODE_SUPPORTED = ['training', 'inference']

# global variable: device
device = None

if torch.cuda.is_available():
    logger.debug(f"Use device: GPU")
    device = torch.device("cuda")
else:
    logger.debug(f"Use device: CPU")
    device = torch.device("cpu")


def _module_from_package(package):
    return package.capitalize() + 'DENVER'


_packages = [
    'denver',
]

for package in _packages:
    module = _module_from_package(package)
    try:
        exec(f'from denver.{package}.model import {module}')
        __all__.append(module)
    except Exception as e:
        pass

for module in __all__:
    logger.info(f'Imported {module}')

# Cleaning imported names
del package, module
