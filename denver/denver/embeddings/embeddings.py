# -*- coding: utf-8
# Copyright (c) 2021 by phucpx@ftech.ai

import os
import logging

from pathlib import Path
from typing import Union, List
from flair.data import Corpus, Dictionary
from flair.models import LanguageModel
from flair.trainers.language_model_trainer import LanguageModelTrainer, TextCorpus
from flair.embeddings import FlairEmbeddings, PooledFlairEmbeddings, StackedEmbeddings

from denver.utils.utils import download_url as dl

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

__embedding_types__ = [
    'flair_embeddings', 
    'bi-flair_embeddings',
    'pooled_flair_embeddings', 
    'bi-pooled_flair_embeddings',
]

__pooling__ = ['min', 'max', 'fade', 'mean']

class Embeddings(object):
    """A Embedding class. """
    def __init__(
        self,
        embedding_types: str='pooled_flair_embeddings',
        pretrain: Union[str, List[str]]=None,
        pooling: str="min",
    ):
        """Initialize a Embeddings class

        :param embedding_types: The type of embedding
        :param pretrain: Path to model pretrained or name pretrained model 
                         Exaples: 'multi-forward', 'multi-backward', 'vi-forward-1024-babe', ...
        :param pooling: The type of pooling used for embedding type is pooled_flair_embeddings. 
                        ('min', 'max', 'fade', 'mean)
        """

        aws_path: str = "http://minio.dev.ftech.ai/resources-denver-v0.0.1-e2d5b5b7/"
        self.PRETRAINED_MODEL_MAP = {
            'vi-forward-1024-wiki': f"{aws_path}vi-forward-1024-wiki.pt",
            'vi-backward-1024-wiki': f"{aws_path}vi-backward-1024-wiki.pt",
            'vi-forward-1024-babe': f"{aws_path}vi-forward-1024-babe.pt",
            'vi-backward-1024-babe': f"{aws_path}vi-backward-1024-babe.pt",
            'vi-forward-1024-lowercase-wiki': f"{aws_path}vi-forward-1024-lowercase-wiki.pt",
            'vi-backward-1024-lowercase-wiki': f"{aws_path}vi-backward-1024-lowercase-wiki.pt",
            'vi-forward-1024-lowercase-babe': f"{aws_path}vi-forward-1024-lowercase-babe.pt",
            'vi-backward-1024-lowercase-babe': f"{aws_path}vi-backward-1024-lowercase-babe.pt",
            'vi-forward-2048-lowercase-wiki': f"{aws_path}vi-forward-2048-lowercase-wiki.pt",
            'vi-backward-2048-lowercase-wiki': f"{aws_path}vi-backward-2048-lowercase-wiki.pt",
        }

        self.embedding_types = embedding_types
        self.pooling = pooling

        if self.pooling not in __pooling__:
            raise ValueError(f"[ERROR] Not support `pooling`: {pooling}."
                            f"The `pooling` value must be in [{', '.join(__pooling__)}]")

        self.dest = './models/.denver/'
        
        if type(pretrain) == str:
            if pretrain.lower() in self.PRETRAINED_MODEL_MAP:
                url = self.PRETRAINED_MODEL_MAP[pretrain.lower()]
                name = pretrain + '.pt'
                dl(url, self.dest, name)
                self.pretrain = os.path.abspath(self.dest + name)
            else:
                self.pretrain = pretrain
            logger.debug(f"\t List pretrained embeddings: {self.pretrain}")
        elif type(pretrain) == list:
            tmps = []
            for item in pretrain:
                if item.lower() in self.PRETRAINED_MODEL_MAP:
                    url = self.PRETRAINED_MODEL_MAP[item.lower()]
                    name = item + '.pt'
                    dl(url, self.dest, name)
                    tmps.append(os.path.abspath(self.dest + name))
                else:
                    tmps.append(item)
            self.pretrain = tmps
            logger.debug(f"\tList pretrained embeddings: {self.pretrain}")
        else:
            raise ValueError()
    def embed(self):
        """ Function embedding

        :returns: embedding: A Embedding as a class embeddings in flair.
        """
        if self.embedding_types not in __embedding_types__:
            raise ValueError(f"\n[ERROR] Not support embedding types: `{self.embedding_types}`. "
                                f"Please select among types: [{', '.join(__embedding_types__)}]")

        logger.debug(f"\tEmbedding types: {self.embedding_types}")

        if self.embedding_types == 'flair_embeddings':
            if self.pretrain is None:
                self.pretrain = 'multi-forward'
            self.embedding = FlairEmbeddings(self.pretrain)
        elif self.embedding_types == 'pooled_flair_embeddings':
            if self.pretrain is None:
                self.pretrain = 'multi-forward'
            self.embedding = PooledFlairEmbeddings(self.pretrain)
        elif self.embedding_types == 'bi-flair_embeddings':
            if self.pretrain is None:
                self.pretrain = ['vi-forward-1024-lowercase-babe', 'vi-backward-1024-lowercase-babe']
            if not isinstance(self.pretrain, List):
                raise ValueError(
                    f"Embedding types: {self.embedding_types} needs to `pretrain` is a List.")
            self.embedding = StackedEmbeddings([FlairEmbeddings(self.pretrain[0]), 
                                                FlairEmbeddings(self.pretrain[1])])
        elif self.embedding_types == 'bi-pooled_flair_embeddings':
            if self.pretrain is None:
                self.pretrain = ['vi-forward-1024-lowercase-babe', 'vi-backward-1024-lowercase-babe']
            if not isinstance(self.pretrain, List):
                raise ValueError(
                    f"Embedding types: {self.embedding_types} needs to `pretrain` is a List.")
            self.embedding = StackedEmbeddings([PooledFlairEmbeddings(self.pretrain[0], pooling=self.pooling),
                                                PooledFlairEmbeddings(self.pretrain[1], pooling=self.pooling)])

        return self.embedding

    def fine_tuning(
        self, 
        corpus_dir: Union[str, Path]='./data/corpus', 
        model_dir: Union[str, Path]='./models',
        dictionary: Union[str, Path]=None,
        is_forward_lm: bool=True, 
        hidden_size: int=1024,
        nlayers: int=1,
        embedding_size: int=100, 
        dropout: float=0.1,
        sequence_length: int=250, 
        batch_size: int=32, 
        learning_rate: float=20, 
        max_epoch: int=500, 
        patience: int=10,
        checkpoint: bool=False
    ):
        """
        :param corpus_dir: Path to corpus folder. the architecture of folder is:
                            corpus/
                            corpus/train/
                            corpus/train/train_split_1
                            corpus/train/train_split_2
                            corpus/train/...
                            corpus/train/train_split_X
                            corpus/test.txt
                            corpus/valid.txt
        :param model_dir: The folder to save checkpoint
        :param dictionary: The name or path to the use dictionary
        :param is_forward_lm: If True, use the language model forward
        :param hidden_size: The number of hidden states in RNN, use when fine-tune without pretrain
        :param rnn_layers: The number of RNN layers, use when fine-tune without pretrain
        :param embedding_size: The size of embedding, use when fine-tune without pretrain
        :param dropout: dropout probability, use when fine-tune without pretrain
        :param sequence_length: Max sequence length
        :param batch_size: Batch size
        :param learning_rate: Learning rate (default=20)
        :param max_epoch: Max epoch (default=20)
        :param patience: Patience is the number of epochs with no improvement the Trainer waits
                         until annealing the learning rate
        :param checkpoint: If True, save checkpoint each epoch when training.

        :returns: embedding: A Embedding as a class embeddings in flair.
        """
        corpus_dir = os.path.abspath(corpus_dir)
        model_dir = os.path.abspath(model_dir)

        if self.embedding_types == 'flair_embeddings' or self.embedding_types == 'pooled_flair_embeddings':
            if self.pretrain is not None:
                language_model = FlairEmbeddings(self.pretrain).lm
                # get the dictionary from the existing language model
                dictionary: Dictionary = language_model.dictionary
                # are you training a forward or backward LM?
                is_forward_lm = language_model.is_forward_lm

                corpus = TextCorpus(corpus_dir,
                            dictionary,
                            is_forward_lm,
                            character_level=True)
            else:
                # load the dictionary
                if dictionary == 'chars' or dictionary == 'common-chars':
                    dictionary : Dictionary = Dictionary.load('chars')
                elif dictionary == "chars-large" or dictionary == "common-chars-large":
                    dictionary : Dictionary = Dictionary.load('chars-large')
                elif dictionary == "chars-xl" or dictionary == "common-chars-xl":
                    dictionary : Dictionary = Dictionary.load('chars-xl')
                elif dictionary == "vi-lowercase-chars":
                    url = f"http://minio.dev.ftech.ai/resources-denver-v0.0.1-e2d5b5b7/vi-uncase-chars?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minio%2F20200825%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200825T183934Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=087f7b02b267256cd74d53f5034e169179b49cbd0e157ee6f476a20c6a4ca46e"

                    dest = './.denver/'
                    name = 'vi-uncase-chars'
                    dl(url, dest, name)
                    path = os.path.abspath(dest + name)
                    dictionary: Dictionary = Dictionary.load_from_file(path)
                elif dictionary == "vi-chars":
                    url = 'http://minio.dev.ftech.ai/resources-denver-v0.0.1-e2d5b5b7/vi-chars?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minio%2F20200825%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200825T184032Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=7922d085c32c2064a769f71e55302eb936c553ba8286fdfb1b97f455739e4553'

                    dest = './.denver/'
                    name = 'vi-chars'
                    dl(url, dest, name)
                    path = os.path.abspath(dest + name)
                    dictionary: Dictionary = Dictionary.load_from_file(path)
                
                corpus = TextCorpus(corpus_dir, 
                                    dictionary, 
                                    is_forward_lm,
                                    character_level=True)
                
                language_model = LanguageModel(dictionary,
                                               is_forward_lm,
                                               hidden_size=hidden_size,
                                               nlayers=nlayers,
                                               embedding_size=embedding_size,
                                               dropout=dropout)

            trainer = LanguageModelTrainer(language_model, corpus)
            trainer.train(model_dir,
                        sequence_length=sequence_length,
                        mini_batch_size=batch_size,
                        learning_rate=learning_rate,
                        max_epochs=max_epoch,
                        patience=patience,
                        checkpoint=checkpoint)

            if self.embedding_types == 'flair_embeddings':
                self.embedding = FlairEmbeddings(trainer.model)
            elif self.embedding_types == 'pooled_flair_embeddings':
                self.embedding = PooledFlairEmbeddings(FlairEmbeddings(trainer.model))
        else:
            raise KeyError(f"Embedding must be a language model."
                           f"Types of embedding are supported: ['flair_embeddings', 'pooled_flair_embeddings']")


        return self.embedding

