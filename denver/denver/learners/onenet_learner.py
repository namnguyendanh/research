# -*- coding: utf-8
# Copyright (c) 2021 by Phuc Phan

import os
import re
import copy
import torch
import shutil
import denver
import logging
import tempfile
import numpy as np

from tqdm import tqdm
from pathlib import Path
from pprint import pformat
from pandas import DataFrame
from scipy.special import softmax
from typing import Union, Iterable

from allennlp.common import Params
from allennlp.data import Instance
from allennlp.common.tqdm import Tqdm
from allennlp.data import DatasetReader
from allennlp.nn import util as nn_util
from allennlp.data.tokenizers import Token
from allennlp.data.iterators import DataIterator
from allennlp.common.checks import check_for_gpu
from allennlp.models.archival import load_archive
from allennlp.common.util import prepare_environment
from allennlp.models.model import Model, _DEFAULT_WEIGHTS
from allennlp.training.util import create_serialization_dir
from allennlp.training.trainer import Trainer, TrainerPieces
from allennlp.models.archival import archive_model, CONFIG_NAME

from denver.constants import *
from denver.models import onenet 
from denver import MODE_SUPPORTED
from denver.data import normalize
from denver.data import DenverDataSource
from denver.learners import DenverLearner
from denver.utils.utils import convert_to_BIO

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class OnenetLearner(DenverLearner):
    """A class OnenetLearner. """
    def __init__(
        self, 
        data_source: DenverDataSource=None, 
        mode: str='training', 
        model_path: str=None, 
        dropout: float=0.5, 
        rnn_type: str='lstm', 
        bidirectional: bool=True, 
        hidden_size: int=200,
        num_layers: int=2,
        word_embedding_dim: int=50,
        word_pretrained_embedding: str='vi-glove-50d', 
        char_embedding_dim: int=30, 
        char_encoder_type: str='cnn',
        num_filters: int=128, 
        ngram_filter_sizes: list=[3],
        conv_layer_activation: str='relu', 
        device: int=0,
        ):
        """ Initialize a OneNet Learner

        :param data_source: A DenverDataSource class
        :param mode: Mode in {inference, training}
        :param model_path: The path to model, (mode = inference)
        :param dropout: dropout probability
        :param rnn_type: The type of Recurrent network
        :param bidirectional: If True, using bidirectional RNN, otherwise, use unidirectional RNN
        :param hidden_size: The number of hidden states in RNN
        :param num_layers: The number of RNN layers
        :param word_embedding_dim: The number dimension of word embedding
        :param word_pretrained_embedding: The name of pretrained word embedding 
                                          or the path to the pretrained word embeding (glove)
        :param char_embedding_dim: The number dimension of char embedding
        :param char_encoder_type: The type char encoder
        :param num_filters: The number of filters (with char_encoder_type='cnn')
        :param ngram_filter_sizes: The list numbers ngram filter sizes
        :param conv_layer_activation: The type activation of conv layer
        :param device: If > using cuda:<device>, else use cpu

        """
        super(OnenetLearner, self).__init__()

        # Verify mode option
        if mode not in MODE_SUPPORTED:
            raise ValueError(
                f"Not support mode: {mode}. "
                f"Please select 1 among these support modes: {''.join(MODE_SUPPORTED)}")

        self.mode = mode
        self.device = device
        self.tempdir = tempfile.mkdtemp()
        self.data_source = data_source
        self.cuda_device = self.device if str(denver.device) == "cuda" else -1
        
        if self.mode.lower() == INFERENCE_MODE:
            if model_path is None:
                logger.error(
                    f"MODE: `inference`, but `model_path` is None value.")
            else:
                model_path = os.path.abspath(model_path)
            
            if os.path.exists(model_path):
                logger.debug(f"Load the model from path: {model_path}")
                self._load_model(model_path)
            else:
                raise ValueError(f"`{model_path}` not supplied or not found.")
        else:
            self.PRETRAINED_WORD_EMBEDDINGs = {
                'vi-glove-50d': 'http://minio.dev.ftech.ai/resources-denver-v0.0.1-e2d5b5b7/viglove_50D.txt', 
                'vi-glove-100d': 'http://minio.dev.ftech.ai/resources-denver-v0.0.1-e2d5b5b7/viglove_100D.txt',
            }

            self.configs = {}
            if data_source.train:
                train_path = os.path.abspath(self.tempdir + '/' + 'train.csv')
                data_source.train.data.to_csv(train_path, encoding='utf-8', index=False)
            
            if data_source.test:
                test_path = os.path.abspath(self.tempdir + '/' + 'test.csv')
                data_source.test.data.to_csv(test_path, encoding='utf-8', index=False)

            if word_pretrained_embedding.lower() not in self.PRETRAINED_WORD_EMBEDDINGs:
                raise ValueError(f"{word_pretrained_embedding} not supported !")
            else:
                
                self.word_pretrained_embedding = self.PRETRAINED_WORD_EMBEDDINGs[word_pretrained_embedding.lower()]
                logger.debug(
                    f"Use word pretrained embedding: {word_pretrained_embedding} - {self.word_pretrained_embedding}")
            
            input_size = word_embedding_dim + num_filters
            lowercase = data_source.lowercase

            self.configs["dataset_reader"] = {
                "type": "onenet",
                "token_indexers": {
                    "tokens": {
                        "type": "single_id",
                        "lowercase_tokens": lowercase
                    },
                    "token_characters": {
                        "type": "characters",
                        "min_padding_length": 3
                    }
                }
            }
            if data_source.train:
                self.configs["train_data_path"] = train_path
            if data_source.test:
                self.configs["test_data_path"] = test_path
            self.configs["model"] = {
                "type": "onenet",
                "label_encoding": "BIO",
                "dropout": dropout,
                "include_start_end_transitions": False,
                "text_field_embedder": {
                    "token_embedders": {
                        "tokens": {
                            "type": "embedding",
                            "embedding_dim": word_embedding_dim,
                            "pretrained_file": self.word_pretrained_embedding,
                            "trainable": True
                        },
                        "token_characters": {
                            "type": "character_encoding",
                            "embedding": {
                                "embedding_dim": char_embedding_dim
                            },
                            "encoder": {
                                "type": char_encoder_type,
                                "embedding_dim": char_embedding_dim,
                                "num_filters": num_filters,
                                "ngram_filter_sizes": ngram_filter_sizes,
                                "conv_layer_activation": conv_layer_activation
                            }
                        }
                    }
                },
                "encoder": {
                    "type": "lstm",
                    "input_size": input_size,
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "bidirectional": bidirectional
                },
                "regularizer": [
                    [
                        "scalar_parameters",
                        {
                        "type": "l2",
                        "alpha": 0.1
                        }
                    ]
                ]
            }

    @property
    def __name__(self):
        return 'OneNet'

    def train(
        self, 
        base_path: str='./models/', 
        is_save_best_model: bool=True,
        model_file: str=None, 
        optimizer: str='adam', 
        batch_size: int=64, 
        learning_rate: float=0.001, 
        num_epochs: int=50, 
        grad_norm: float=5.0, 
        patience: int=75, 
        skip_save_model: bool=False, 
        skip_save_log: bool=True, 
        **kwargs
    ):
        """Train and save the model

        :param base_path: Main path to models are saved
        :param is_save_best_model: If True save the best model, else save the final model
        :param model_file: The file name to save the model
        :param learning_rate: Initial learning rate
        :param batch_size: Size of batches during training
        :param num_epochs: The number of epochs to train.
        :param grad_norm: If provided, gradient norms will be rescaled to have a maximum of this value.
        :param patience: Number of epochs to be patient before early stopping: the training is stopped 
                        after patience epochs with no improvement. If given, it must be > 0. If None, 
                        early stopping is disabled.
        :param skip_save_model: If True, disables saving the model
        :param skip_save_log: If True, disable saving the training log
        """
        self.configs["iterator"] = {
            "type": "basic",
            "batch_size": batch_size
        }
        self.configs["trainer"] = {
            "optimizer": {
                "type": optimizer,
                'lr': learning_rate,
            },
            "validation_metric": "+main_score", 
            "num_serialized_models_to_keep": 3,
            "num_epochs": num_epochs,
            "grad_norm": grad_norm, 
            "patience": patience, 
            "cuda_device": self.cuda_device
        }
        
        self.config = Params(self.configs)
        _config = copy.deepcopy(self.config)
    
        # clean folder base_path
        if os.path.exists(base_path):
            shutil.rmtree(os.path.abspath(base_path))

        prepare_environment(_config)
        create_serialization_dir(_config, serialization_dir=base_path, recover=False, force=False)
        check_for_gpu(self.cuda_device)

        copy_config = copy.deepcopy(self.config)
        copy_config.to_file(os.path.join(base_path, CONFIG_NAME))

        dataset_reader_params = copy_config["dataset_reader"]
        self.dataset_reader = DatasetReader.from_params(dataset_reader_params)
    
        self.pieces = TrainerPieces.from_params(_config, serialization_dir=base_path, recover=False)  # pylint: disable=no-member
        trainer = Trainer.from_params(
                model=self.pieces.model,
                serialization_dir=base_path,
                iterator=self.pieces.iterator,
                train_data=self.pieces.train_dataset,
                validation_data=self.pieces.validation_dataset,
                params=self.pieces.params,
                validation_iterator=self.pieces.validation_iterator)

        try:
            _ = trainer.train()
        except KeyboardInterrupt:
            # if we have completed an epoch, try to create a model archive.
            if os.path.exists(os.path.join(base_path, _DEFAULT_WEIGHTS)):
                logging.info(
                    f"Training interrupted by the user. Attempting to create "
                    f"a model archive using the current best epoch weights."
                )
                archive_model(base_path, files_to_archive=_config.files_to_archive)
            raise
        
        # # Now tar up results
        if not skip_save_model:
            archive_model(base_path, model_file=model_file, files_to_archive=_config.files_to_archive)
            logger.info(f"Path to the saved model: {Path(base_path)/model_file}")

        self.model = trainer.model


        ## remove trash file
        shutil.rmtree(Path(base_path)/'log', ignore_errors=True)

        for f in os.listdir(Path(base_path)):
            if re.search(r"model_state_epoch_([0-9\.\-]+)\.th", f):
                os.remove(os.path.join(Path(base_path)/f))

            if re.search(r"training_state_epoch_([0-9\.\-]+)\.th", f):
                os.remove(os.path.join(Path(base_path)/f))
        
            if skip_save_log:
                if re.search(r"metrics_epoch_([0-9\.\-]+)\.json", f):
                    os.remove(os.path.join(Path(base_path)/f))
        

    def _load_model(self, model_path):
        """Load pretrained model

        :param model_path: Path to the pretrained model
        """
        archive = load_archive(model_path, cuda_device=self.cuda_device)
        self.config = copy.deepcopy(archive.config)
        dataset_reader_params = archive.config["dataset_reader"]
        self.dataset_reader = DatasetReader.from_params(dataset_reader_params)
        self.model = archive.model
        self.model.eval()

    def predict(
        self, 
        sample: str=None, 
        lowercase: bool=False, 
        rm_emoji: bool=True, 
        rm_url: bool=True, 
        rm_special_token: bool=False
    ):
        """Predicts the output for the given sequence sample

        :param sample: The sample
        :param lowercase: If True, lowercase data
        :param rm_emoji: If True, replace the emoji token into <space> (" ")
        :param rm_url: If True, replace the url token into <space> (" ")
        :param rm_special_token: If True, replace the special token into <space> (" "), 
                                 special token included of punctuation token, characters without vietnamese characters

        :returns: Return the output
        """
        self.model.eval()
        sample = normalize(sample, rm_emoji=rm_emoji, rm_url=rm_url, 
                           rm_special_token=rm_special_token, lowercase=lowercase)

        if len(sample) == 0:
            return None

        tokens = [Token(s) for s in sample.split()]
        instance = self.dataset_reader.text_to_instance(tokens)
        outputs = self.model.forward_on_instance(instance)

        return outputs

    def convert_to_rasa_format(self, outputs):
        """
        Convert outputs in rasa's format
        
        :param outputs: Dict output from the output of predict function.

        :returns: A result rasa format. Example: \n
                    [{  \n
                        'confidence': confidence, \n 
                        'end': end, \n
                        'start': start, \n
                        'entity': entity, \n
                        'extractor': extractor, \n
                        'value': value  \n
                    }] \n
        """

        rasa_output = {}

        if not outputs:
            return {}

        rasa_output["text"] = " ".join(outputs.get("words", []))
        rasa_output["intent"] = {
            "name": outputs.get("intent"),
            "confidence": max(outputs.get("intent_probs")).item(),
            "intent_logits": outputs.get("intent_logits", None)
        }

        tags = outputs.get('tags')
        words = outputs.get('words')
        tag_probs = softmax(outputs.get('tag_logits'), axis=1)
        tag_maxs = np.amax(tag_probs, axis=1)

        # get index start words
        ids = [0]
        temp = 0
        for i in range(1, len(words)):
            ids.append(temp + len(words[i-1]) + 1)
            temp = ids[-1]
        ids.append(len(rasa_output["text"]) + 1)

        entities = []
        start = 0
        entity = None
        end = 0
        ping = False

        for i in range(len(tags)):
            if ping == True:
                if tags[i] == 'O':
                    end = i
                    entities.append({
                        'entity': entity, 
                        'start': ids[start], 
                        'end': ids[end] - 1,                     
                        'value': ' '.join(words[start:end]).strip(),
                        'confidence': np.average(tag_maxs[start:end]).item(),
                        'extractor': self.__name__
                    })
                    ping = False

                elif ("B-" in tags[i]) and (i == len(tags) - 1):
                    end = i
                    entities.append({
                        'entity': entity, 
                        'start': ids[start], 
                        'end': ids[end] - 1,                     
                        'value': ' '.join(words[start:end]).strip(),
                        'confidence': np.average(tag_maxs[start:end]).item(),
                        'extractor': self.__name__
                    })

                    start = i
                    end = i + 1
                    entity = tags[i][2:]

                    entities.append({
                        'entity': entity, 
                        'start': ids[start], 
                        'end': ids[end] - 1,
                        'value': ' '.join(words[start:end]).strip(),
                        'confidence': np.average(tag_maxs[start:end]).item(),
                        'extractor': self.__name__
                    })

                elif "B-" in tags[i]:
                    end = i
                    entities.append({
                        'entity': entity, 
                        'start': ids[start], 
                        'end': ids[end] - 1,                     
                        'value': ' '.join(words[start:end]).strip(),
                        'confidence': np.average(tag_maxs[start:end]).item(),
                        'extractor': self.__name__
                    })
                    ping = True
                    start = i
                    entity = tags[i][2:]

                elif i == len(tags) - 1:
                    end = i + 1
                    entities.append({
                        'entity': entity, 
                        'start': ids[start], 
                        'end': ids[end] - 1,
                        'value': ' '.join(words[start:end]).strip(),
                        'confidence': np.average(tag_maxs[start:end]).item(),
                        'extractor': self.__name__
                    })

            else:
                if "B-" in tags[i] and i == len(tags) - 1:
                    start = i
                    end = i + 1
                    entity = tags[i][2:]
                    entities.append({
                        'entity': entity, 
                        'start': ids[start], 
                        'end': ids[end] - 1,
                        'value': ' '.join(words[start:end]).strip(),
                        'confidence': np.average(tag_maxs[start:end]).item(),
                        'extractor': self.__name__
                    })

                elif "B-" in tags[i]:
                    start = i
                    entity = tags[i][2:]
                    ping = True

        rasa_output["entities"] = entities

        return rasa_output

    def process(
        self, 
        sample: str=None, 
        lowercase: bool=False, 
        rm_emoji: bool=True, 
        rm_url: bool=True, 
        rm_special_token: bool=False
    ):
        """Return the results as output of rasa format

        :param sample: The sample need to inference
        :param lowercase: If True, lowercase data
        :param rm_emoji: If True, replace the emoji token into <space> (" ")
        :param rm_url: If True, replace the url token into <space> (" ")
        :param rm_special_token: If True, replace the special token into <space> (" "),
                                 special token included of punctuation token, characters without vietnamese characters

        :returns: The results format rasa. Example: \n
                    { \n
                        "intent": { \n
                            "name": query_kb, \n
                            "confidence": 0.9999999\n
                            }, \n
                        "entities": [\n
                           {\n
                                'confidence': 0.9994359612464905,\n
                                'end': 3,\n
                                'entity': 'object_type',\n
                                'extractor': 'OnetNet',\n
                                'start': 0,\n
                                'value': 'cũi'\n
                            }, \n
                        ],\n
                        "text": cũi này còn k sh\n
                     }\n
        """
        self.model.eval()

        outputs = self.predict(sample=sample, lowercase=lowercase, rm_emoji=rm_emoji, 
                            rm_url=rm_url, rm_special_token=rm_special_token)

        logger.debug(f"Raw output: {pformat(outputs)}")
        
        rasa_output = self.convert_to_rasa_format(outputs)

        return rasa_output
        
        
    def _evaluate(
        self, 
        model: Model, 
        instances: Iterable[Instance], 
        data_iterator: DataIterator, 
        cuda_device: int, 
        batch_weight_key: str=""
    ):
        check_for_gpu(cuda_device)
        with torch.no_grad():
            model.eval()

            iterator = data_iterator(instances,
                                    num_epochs=1,
                                    shuffle=False)
            logger.info("Iterating over dataset")
            generator_tqdm = Tqdm.tqdm(iterator, total=data_iterator.get_num_batches(instances))

            # Number of batches in instances.
            batch_count = 0
            # Number of batches where the model produces a loss.
            loss_count = 0
            # Cumulative weighted loss
            total_loss = 0.0
            # Cumulative weight across all batches.
            total_weight = 0.0

            for batch in generator_tqdm:
                batch_count += 1
                batch = nn_util.move_to_device(batch, cuda_device)
                output_dict = model(**batch)
                loss = output_dict.get("loss")

                metrics = model.get_metrics()

                if loss is not None:
                    loss_count += 1
                    if batch_weight_key:
                        weight = output_dict[batch_weight_key].item()
                    else:
                        weight = 1.0

                    total_weight += weight
                    total_loss += loss.item() * weight
                    # Report the average loss so far.
                    metrics["loss"] = total_loss / total_weight

            final_metrics = model.get_metrics(reset=True, mode=True)
            if loss_count > 0:
                # Sanity check
                if loss_count != batch_count:
                    raise RuntimeError("The model you are trying to evaluate only sometimes " +
                                    "produced a loss!")
                final_metrics["loss"] = total_loss / total_weight

        return final_metrics

    def get_classes(self):
        """Get all classes of intent and tags
        
        :returns: A dict {'intent': classes of intent, 'tag': classes of tag}
        """

        cls_classes = self.model.vocab._index_to_token['intent_labels']
        tag_classes = self.model.vocab._index_to_token['labels']
        return {'intent': cls_classes, 'tag': tag_classes}

    def evaluate(
        self, 
        data: Union[str, DataFrame, DenverDataSource]=None, 
        text_col: str='text', 
        intent_col: str='intent', 
        tag_col: str='tag',
        lowercase: bool=False, 
        rm_emoji: bool=True, 
        rm_url: bool=True, 
        rm_special_token: bool=False, 
        **kwargs
    ):
        """Evaluate the model with test dataset

        :param data: The path to data .csv or a DataFrame or a DenverDataSource
        :param text_col: The column name of text data
        :param intent_col: The column name of intent label data
        :param tag_col: The column name of tags label data
        :param lowercase: If True, lowercase data
        :param rm_emoji: If True, replace the emoji token into <space> (" ")
        :param rm_url: If True, replace the url token into <space> (" ")
        :param rm_special_token: If True, replace the special token into <space> (" "), 
                                 special token included of punctuation token, characters without vietnamese characters

        :returns: Metrics format: \n 
                    {\n
                        'precision': 1.0, \n
                        'recall': 1.0, \n
                        'f1-score': 1.0 \n
                    } \n
        """
        tempdir = tempfile.mkdtemp()
        if isinstance(data, str):
            data_path = os.path.abspath(data)
            data_source = DenverDataSource.from_csv(test_path=data_path,
                                                    text_col=text_col,
                                                    intent_col=intent_col,
                                                    tag_col=tag_col,
                                                    rm_emoji=rm_emoji,
                                                    rm_url=rm_url,
                                                    rm_special_token=rm_special_token,
                                                    lowercase=lowercase)
            data_path = os.path.abspath(tempdir + '/' + 'data.csv')
            data_source.test.data.to_csv(data_path, encoding='utf-8', index=False)
        elif isinstance(data, DataFrame):
            data_path = os.path.abspath(tempdir + '/' + 'data.csv')
            data_source = DenverDataSource.from_df(test_df=data,
                                             text_col=text_col,
                                             intent_col=intent_col,
                                             tag_col=tag_col,
                                             rm_emoji=rm_emoji,
                                             rm_url=rm_url,
                                             rm_special_token=rm_special_token,
                                             lowercase=lowercase)
            data_source.test.data.to_csv(data_path, encoding='utf-8', index=False)
        elif isinstance(data, DenverDataSource):
            data_path = os.path.abspath(tempdir + '/' + 'data.csv')
            if data.test:
                data.test.data.to_csv(data_path, encoding='utf-8', index=False)
            else:
                raise ValueError(f"Attribute `test` in `data` ({type(data)}) is `None` value!")
        else:
            raise ValueError(f"`data` must be difference to `None` value.")
        
        config = copy.deepcopy(self.config)

        prepare_environment(config)
        self.model.eval()
        dataset_reader = DatasetReader.from_params(config.pop('dataset_reader'))

        logger.info("Reading evaluation data from %s", data_path)
        instances = dataset_reader.read(data_path)

        iterator_params = config.pop("validation_iterator", None)
        if iterator_params is None:
            iterator_params = config.pop("iterator")
        iterator = DataIterator.from_params(iterator_params)
        iterator.index_with(self.model.vocab)

        logger.info(f"Evaluating...")
        metrics = self._evaluate(model=self.model, 
                                 instances=instances, 
                                 data_iterator=iterator, 
                                 cuda_device=self.cuda_device, 
                                 batch_weight_key="")

        metrics = {
            'loss': [round(metrics['loss'], 4)],
            'main_score': [round(metrics['main_score'], 4)],
            'intent': [metrics['intent']],
            'tags': [metrics['tags']],
            'cls_detailed': metrics['cls_detailed'],
            'tags_detailed': metrics['tags_detailed']
        }

        return metrics

    def validate(self):
        evaluation_iterator = self.pieces.validation_iterator or self.pieces.iterator
        evaluation_dataset = self.pieces.validation_dataset
        # Evaluate
        # logging.getLogger('allennlp.training.util').disabled = True
        metrics = None 
        if evaluation_dataset:
            metrics = self._evaluate(self.model, evaluation_dataset, evaluation_iterator,
                                    cuda_device=self.cuda_device, batch_weight_key="")
        if metrics:
            metrics = {
                'loss': [round(metrics['loss'], 4)],
                'main_score': [round(metrics['main_score'], 4)],
                'intent': [metrics['intent']],
                'tags': [metrics['tags']],
            }
            return metrics
            
        return None

    def predict_on_df(
        self, 
        data: Union[str, Path, DataFrame], 
        text_col: str='text',
        intent_col: str=None,
        tag_col: str=None,
        lowercase: bool=True, 
        rm_emoji: bool=True, 
        rm_url: bool=True, 
        rm_special_token: bool=False,
        **kwargs 
    ):
        """Predicts the class labels for a DataFrame
        
        :param data: The path to data or a DataFrame
        :param text_col: The column name of text data
        :param intent_col: The column name of intent label data
        :param tags_col: The column name of tags label data
        :param lowercase: If True, lowercase data
        :param rm_emoji: If True, replace the emoji token into <space> (" ")
        :param rm_url: If True, replace the url token into <space> (" ")
        :param rm_special_token: If True, replace the special token into <space> (" "),
                                 special token included of punctuation token, characters without vietnamese characters

        :returns: A DataFrame include of columns 'text', 'pred', 'label'(maybe)). 

        """
        if isinstance(data, str) or isinstance(data, Path):
            data_source = DenverDataSource.from_csv(test_path=data,
                                                    text_col=text_col, 
                                                    intent_col=intent_col,
                                                    tag_col=tag_col, 
                                                    rm_emoji=rm_emoji, 
                                                    rm_url=rm_url, 
                                                    rm_special_token=rm_special_token, 
                                                    lowercase=lowercase)
            data_df = data_source.test.data
        else:
            data_df = data
        
        if 'text' not in data_df.columns:
            raise ValueError(f"Column name in DataFrame must be include `text` column.")
        
        tags_preds = []
        intent_preds = []
        intent_scores = []

        logger.info(f"Get-prediction...")
        for i in tqdm(range(len(data_df))):
            output = self.process(
                            sample=data_df['text'][i], 
                            rm_emoji=rm_emoji, 
                            rm_url=rm_url, 
                            rm_special_token=rm_special_token, 
                            lowercase=lowercase
                        )

            bio = convert_to_BIO(entities=output['entities'], text=data_df['text'][i])
            tags_preds.append(bio)
            intent_preds.append(output['intent']['name'])
            intent_scores.append(output['intent']['confidence'])
        
        data_df['intent_pred'] = intent_preds
        data_df['intent_confidence'] = intent_scores
        data_df['tag_pred'] = tags_preds

        return data_df
