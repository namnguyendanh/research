# -*- coding: utf-8
# Copyright (c) 2021 by phucpx@ftech.ai

import logging

from denver.constants import *

logger = logging.getLogger(__name__)


ulmfit_hiperot_config_default = {
    EXECUTOR: {TYPE: SERIAL}, 
    GOAL: MAXIMIZE, 
    PARAMETERS: {
                    'drop_mult': {
                        'range': [0.2, 1.0], 
                        'space': 'log', 
                        'type': 'float', 
                        }, 
                    'learning_rate': {
                        'range': [1e-3, 1e-1], 
                        'space': 'log', 
                        'type': 'float', 
                        }, 
                    'num_epochs': {
                        'range': [10, 30], 
                        'space': 'linear', 
                        'type': 'int', 
                    }
                }, 
    SAMPLER: {TYPE: RANDOM_SAMPLER, NUM_SAMPLES: 3}, 
    METRIC: ACCURACY,
    RUN: {
        'early_stopping_epochs': False, 
    }

}

onenet_hiperopt_config_default = {
    EXECUTOR: {NUM_WORKERs: 4, TYPE: PARALLEL}, 
    GOAL: MAXIMIZE, 
    PARAMETERS: {
        'learning_rate': {
            'range': [1e-4, 1e-2], 
            'type': 'float', 
            'space': 'log'
        }, 
        'batch_size': {
            'range': [64, 128], 
            'space': 'linear', 
            'type': 'int', 
        },
        'dropout': {
            'range': [0.2, 0.8], 
            'type': 'float', 
            'space': 'log'
        },
        'hidden_size': {
            'range': [200, 512], 
            'type': 'int',
            'space': 'linear'
        },
        'char_embedding_dim': {
            'range': [30, 100], 
            'type': 'int',
            'space': 'linear'
        }, 
        'num_filters': {
            'range': [64, 512], 
            'type': 'int', 
            'space': 'linear'
        }

    }, 
    SAMPLER: {TYPE: RANDOM_SAMPLER, NUM_SAMPLES: 10}, 
    METRIC: MAIN_SCORE, 
    RUN: {
        'early_stopping_epochs': 15, 
    }
}

flair_hiperopt_config_default = {
    EXECUTOR: {TYPE: SERIAL}, 
    GOAL: MAXIMIZE,
    PARAMETERS: {
        'hidden_size': {
            'range': [512, 2048], 
            'type': 'int', 
            'space': 'linear'
        },
        'rnn_layers': {
            'range': [1, 2], 
            'type': 'int', 
            'space': 'linear', 
        }, 
        'dropout': {
            'range': [0.0, 0.5], 
            'type': 'float',
            'space': 'log', 
        }, 
        'word_dropout': {
            'range': [0.05, 0.3], 
            'type': 'float', 
            'space': 'log'
        },
        'locked_dropout': {
            'range': [0.4, 0.7],
            'type': 'float', 
            'space': 'log'
        }, 
        'reproject_embeddings': {
            'range': [1024, 4096], 
            'type': 'int', 
            'space': 'linear'
        }, 
        'learning_rate': {
            'range': [0.01, 0.1], 
            'type': 'float', 
            'space': 'log'
        }, 
        'batch_size': {
            'range': [32, 64], 
            'space': 'linear', 
            'type': 'int', 
        }
    }, 
    SAMPLER: {TYPE: RANDOM_SAMPLER, NUM_SAMPLES: 10}, 
    METRIC: MAIN_SCORE,
    RUN: {
        'early_stopping_epochs': 5, 
    }
}

hiperopt_config_default_registry = {
    'ulmfit_classifier': ulmfit_hiperot_config_default, 
    'flair_sequence_tagger': flair_hiperopt_config_default, 
    'onenet': onenet_hiperopt_config_default, 
}