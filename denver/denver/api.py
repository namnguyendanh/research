# -*- coding: utf-8
# Copyright (c) 2021 by phucpx@ftech.ai

import os
import pandas as pd
import logging

from typing import Union, List, Any
from pathlib import Path

from denver import logger
from denver.constants import *
from denver.data.data_source import DenverDataSource
from denver.learners import ULMFITClassificationLearner
from denver.learners import FlairSequenceTaggerLearner
from denver.learners import OnenetLearner
from denver.embeddings.embeddings import Embeddings
from denver.trainers.language_model_trainer import LanguageModelTrainer
from denver.trainers.trainer import ModelTrainer as Trainer
from denver.data import split_data
from denver.utils.utils import ifnone, convert_to_denver_format
from denver.utils.config_parser import get_config_yaml


class DenverAPI:
    def __init__(self, config: Union[str, dict] ) -> None:
        """Constructor for the DenverAPI class. """
        super(DenverAPI, self).__init__()

    @classmethod
    def train(cls, config: Union[str, dict, Path], monitor: bool=False) -> None:

        if isinstance(config, str) or isinstance(config, Path):
            config = get_config_yaml(config_file=config)

        _type = config[MODEL][OUTPUT_FEATURES][TYPE]
        train_path = config[DATASET].get(TRAIN_PATH, None)
        test_path = config[DATASET].get(TEST_PATH, None)
        text_col = config[DATASET].get(TEXT_COL, 'text')
        label_col = config[DATASET].get(LABEL_COL, None)
        intent_col = config[DATASET].get(INTENT_COL, None)
        tag_col = config[DATASET].get(TAG_COL, None)

        pre_processing = config[DATASET].get(PRE_PROCESSING)

        encoder = config[MODEL][INPUT_FEATURES][ENCODER].get(NAME)
        encoder_args = config[MODEL][INPUT_FEATURES][ENCODER][ARGS]

        decoder = config[MODEL][INPUT_FEATURES].get(DECODER)

        training_params = config[TRAINING_PARAMS].get(HYPER_PARAMS)

        base_path = config[TRAINING_PARAMS].get(BASE_PATH)
        is_save_best_model = config[TRAINING_PARAMS].get('is_save_best_model')
        model_file = config[TRAINING_PARAMS].get('model_file')

        balance = pre_processing.get(BALANCE, False)
        size = balance.get('size') if balance else None
        replace = balance.get('replace') if balance else False
        is_lowercase = pre_processing.get(LOWERCASE_TOKEN, True)
        rm_emoji = pre_processing.get(REMOVE_EMOJI, False)
        rm_special_token = pre_processing.get(REMOVE_SPECIAL_TOKEN, False)
        rm_url = pre_processing.get(REMOVE_URL, False)

        logger.debug(
            f"Balance args: balance={balance} | size={size} | replace={replace}")

        data_source = DenverDataSource.from_csv(train_path=train_path,
                                            test_path=test_path,
                                            text_col=text_col,
                                            label_col=label_col, 
                                            intent_col=intent_col,
                                            tag_col=tag_col,
                                            lowercase=is_lowercase,
                                            rm_emoji=rm_emoji,
                                            rm_url=rm_url,
                                            rm_special_token=rm_special_token,
                                            balance=balance,
                                            size=size,
                                            replace=replace)

        if encoder.lower() in ONENET_NLU:
            learn = OnenetLearner(
                mode=TRAINING_MODE,
                data_source=data_source,
                dropout=encoder_args.get('dropout', 0.5),
                rnn_type=encoder_args.get('rnn_type', 'lstm'),
                bidirectional=encoder_args.get('bidirectional', True),
                hidden_size=encoder_args.get('hidden_size', 200),
                num_layers=encoder_args.get('num_layers', 2),
                word_embedding_dim=encoder_args.get('word_embedding_dim', 50),
                word_pretrained_embedding=encoder_args.get('word_pretrained_embedding', 'vi-glove-50d'),
                char_embedding_dim=encoder_args.get('char_embedding_dim', 30),
                char_encoder_type=encoder_args.get('char_encoder_type', 'cnn'),
                num_filters=encoder_args.get('num_filters', 128),
                ngram_filter_sizes=encoder_args.get('ngram_filter_sizes', [3]),
                conv_layer_activation=encoder_args.get('conv_layer_activation', 'relu')
            )

        elif encoder.lower() in ULMFIT_CLASSIFIER:
            lm_pretrain = config[MODEL][INPUT_FEATURES].get('pretrain_language_model', None)

            # Fine-tuning LM from Training Dataset
            lm_trainer = LanguageModelTrainer(pretrain=lm_pretrain)
            lm_trainer.fine_tuning_from_df(data_df=data_source.train.data,
                                        batch_size=training_params.get('batch_size', 128))

            # define model
            learn = ULMFITClassificationLearner(
                mode=TRAINING_MODE, 
                data_source=data_source, 
                drop_mult=encoder_args.get('drop_mult', 0.3), 
                average=encoder_args.get('average', 'weighted'), 
                beta=encoder_args.get('beta', 1)
            )

        elif encoder.lower() in FLAIR_SEQUENCE_TAGGER:
            embedding_types = config[MODEL][INPUT_FEATURES].get('embedding_types', None)
            pretrain_embedding = config[MODEL][INPUT_FEATURES].get('pretrain_embedding', None)

            embeddings = Embeddings(
                embedding_types=embedding_types, pretrain=pretrain_embedding)
            embedding = embeddings.embed()

            learn = FlairSequenceTaggerLearner(
                mode=TRAINING_MODE,
                data_source=data_source,
                tag_type=_type,
                embeddings=embedding,
                hidden_size=encoder_args.get('hidden_size', 1024),
                use_rnn=encoder_args.get('use_rnn', True),
                rnn_layers=encoder_args.get('rnn_layers', 1),
                dropout=encoder_args.get('dropout', 0.0),
                word_dropout=encoder_args.get('word_dropout', 0.05),
                locked_dropout=encoder_args.get('locked_dropout', 0.5),
                reproject_embeddings=encoder_args.get('reproject_embeddings', True),
                use_crf=decoder.get('crf', True),
                beta=encoder_args.get('beta', 1)
            )
                                            
        else:
            raise ValueError(
                f'This `{encoder}` encoder model is not supported.')

        # Get params training model
        batch_size = training_params.get('batch_size')
        learning_rate = float(training_params.get('learning_rate'))
        num_epochs = training_params.get('num_epochs')
        moms = training_params.get('momentums')

        # Training the  model
        trainer = Trainer(learn=learn)
        trainer.train(base_path=base_path,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    num_epochs=num_epochs,
                    monitor_test=monitor,
                    model_file=model_file,
                    moms=moms,
                    is_save_best_model=is_save_best_model)

    @classmethod
    def test(cls, config: Union[str, dict, Path], input: str) -> Any:
        if isinstance(config, str) or isinstance(config, Path):
            config = get_config_yaml(config_file=config)

        _type = config[MODEL][OUTPUT_FEATURES][TYPE]
        
        pre_processing = config[DATASET][PRE_PROCESSING]
        encoder = config[MODEL][INPUT_FEATURES][ENCODER][NAME]

        is_lowercase = pre_processing.get(LOWERCASE_TOKEN, True)
        rm_emoji = pre_processing.get(REMOVE_EMOJI, False)
        rm_special_token = pre_processing.get(REMOVE_SPECIAL_TOKEN, False)
        rm_url = pre_processing.get(REMOVE_URL, False)

        base_path = config[TRAINING_PARAMS]['base_path']
        model_file = config[TRAINING_PARAMS].get('model_file')
        model_path = Path(base_path)/model_file

        if encoder.lower() in ULMFIT_CLASSIFIER:
            learn = ULMFITClassificationLearner(mode=INFERENCE_MODE, model_path=model_path)

        elif encoder.lower() in FLAIR_SEQUENCE_TAGGER:
            learn = FlairSequenceTaggerLearner(mode=INFERENCE_MODE, model_path=model_path)

        elif encoder.lower() in ONENET_NLU:
            learn = OnenetLearner(mode=INFERENCE_MODE, model_path=model_path)

        else:
            raise ValueError(
                f'This `{encoder}` encoder model is not supported.')

        output = learn.process(
                    sample=input, 
                    lowercase=is_lowercase, 
                    rm_emoji=rm_emoji, 
                    rm_url=rm_url, 
                    rm_special_token=rm_special_token
                )

        return output

    @classmethod
    def evaluate(cls, config: Union[str, dict, Path], data_path: Union[str, Path]) -> dict:
        if isinstance(config, str) or isinstance(config, Path):
            config = get_config_yaml(config_file=config)

        _type = config[MODEL][OUTPUT_FEATURES][TYPE]
        
        data_path = ifnone(data_path, config[DATASET].get(TEST_PATH))
        text_col = config[DATASET].get(TEXT_COL, 'text')
        label_col = config[DATASET].get(LABEL_COL, None)
        intent_col = config[DATASET].get(INTENT_COL, None)
        tag_col = config[DATASET].get(TAG_COL, None)

        pre_processing = config[DATASET][PRE_PROCESSING]
        encoder = config[MODEL][INPUT_FEATURES][ENCODER][NAME]
        base_path = config[TRAINING_PARAMS][BASE_PATH]
        model_file = config[TRAINING_PARAMS].get('model_file')
        model_path = Path(base_path)/model_file

        is_lowercase = pre_processing.get(LOWERCASE_TOKEN, True)
        rm_emoji = pre_processing.get(REMOVE_EMOJI, False)
        rm_special_token = pre_processing.get(REMOVE_SPECIAL_TOKEN, False)
        rm_url = pre_processing.get(REMOVE_URL, False)

        if encoder.lower() in FLAIR_SEQUENCE_TAGGER:
            learn = FlairSequenceTaggerLearner(mode=INFERENCE_MODE, 
                                               model_path=model_path)

        elif encoder.lower() in ULMFIT_CLASSIFIER:
            learn = ULMFITClassificationLearner(mode=INFERENCE_MODE, model_path=model_path)

        elif encoder.lower() in ONENET_NLU:
            
            learn = OnenetLearner(mode=INFERENCE_MODE, model_path=model_path)
        else:
            raise ValueError(
                f'This `{encoder}` encoder model is not supported.')

        results = learn.evaluate(
                        data=data_path, 
                        text_col=text_col, 
                        label_col=label_col, 
                        intent_col=intent_col, 
                        tag_col=tag_col, 
                        lowercase=is_lowercase, 
                        rm_emoji=rm_emoji, 
                        rm_url=rm_url, 
                        rm_special_token=rm_special_token
                    )

        return results

    @classmethod
    def experiment(
        cls, 
        config: Union[str, dict, Path], 
        dataset: Union[str, Path, pd.DataFrame]=None, 
        pct: float=0.1, 
        seed: int=123, 
        skip_save_model: bool=False,
        early_stopping_epochs: Union[bool, int]=False, 
        verbose: bool=False
    ) -> None:

        if isinstance(config, str) or isinstance(config, Path):
            config = get_config_yaml(config_file=config)

        _type = config[MODEL][OUTPUT_FEATURES][TYPE]

        train_path = config[DATASET].get(TRAIN_PATH, None)
        test_path = config[DATASET].get(TEST_PATH, None)
        

        text_col = config[DATASET].get(TEXT_COL)
        label_col = config[DATASET].get(LABEL_COL)
        intent_col = config[DATASET].get(INTENT_COL)
        tag_col = config[DATASET].get(TAG_COL)

        pre_processing = config[DATASET][PRE_PROCESSING]
        encoder = config[MODEL][INPUT_FEATURES][ENCODER][NAME]
        encoder_args = config[MODEL][INPUT_FEATURES][ENCODER][ARGS]
        decoder = config[MODEL][INPUT_FEATURES].get(DECODER)

        training_params = config[TRAINING_PARAMS][HYPER_PARAMS]

        base_path = config[TRAINING_PARAMS].get('base_path')
        is_save_best_model = config[TRAINING_PARAMS].get('is_save_best_model')
        model_file = config[TRAINING_PARAMS].get('model_file')

        balance = pre_processing.get(BALANCE, False)
        size = balance.get('size') if balance else None
        replace = balance.get('replace') if balance else False
        is_lowercase = pre_processing.get(LOWERCASE_TOKEN, True)
        rm_emoji = pre_processing.get(REMOVE_EMOJI, False)
        rm_special_token = pre_processing.get(REMOVE_SPECIAL_TOKEN, False)
        rm_url = pre_processing.get(REMOVE_URL, False)

        logger.debug(
            f"Balance args: balance={balance} | size={size} | replace={replace}")

        if dataset is not None:
            if isinstance(dataset, str) or isinstance(dataset, Path):
                dataset = pd.read_csv(dataset, encoding='utf-8')

            train_df, test_df = split_data(dataset, pct=pct, seed=seed)

            data_source = DenverDataSource.from_df(train_df=train_df,
                                                    test_df=test_df,
                                                    text_col=text_col,
                                                    label_col=label_col, 
                                                    intent_col=intent_col,
                                                    tag_col=tag_col,
                                                    lowercase=is_lowercase,
                                                    rm_emoji=rm_emoji,
                                                    rm_url=rm_url,
                                                    rm_special_token=rm_special_token,
                                                    balance=balance,
                                                    size=size,
                                                    replace=replace)
        elif train_path is not None:
            data_source = DenverDataSource.from_csv(train_path=train_path,
                                                    test_path=test_path,
                                                    text_col=text_col,
                                                    label_col=label_col, 
                                                    intent_col=intent_col,
                                                    tag_col=tag_col,
                                                    lowercase=is_lowercase,
                                                    rm_emoji=rm_emoji,
                                                    rm_url=rm_url,
                                                    rm_special_token=rm_special_token,
                                                    balance=balance,
                                                    size=size,
                                                    replace=replace)
        else:
            raise ValueError(f"Experiment dataset is empty !")

        # Training the class model
        if encoder.lower() in ONENET_NLU:
            learn = OnenetLearner(
                mode=TRAINING_MODE,
                data_source=data_source,
                dropout=encoder_args.get('dropout', 0.5),
                rnn_type=encoder_args.get('rnn_type', 'lstm'),
                bidirectional=encoder_args.get('bidirectional', True),
                hidden_size=encoder_args.get('hidden_size', 200),
                num_layers=encoder_args.get('num_layers', 2),
                word_embedding_dim=encoder_args.get('word_embedding_dim', 50),
                word_pretrained_embedding=encoder_args.get('word_pretrained_embedding', 'vi-glove-50d'),
                char_embedding_dim=encoder_args.get('char_embedding_dim', 30),
                char_encoder_type=encoder_args.get('char_encoder_type', 'cnn'),
                num_filters=encoder_args.get('num_filters', 128),
                ngram_filter_sizes=encoder_args.get('ngram_filter_sizes', [3]),
                conv_layer_activation=encoder_args.get('conv_layer_activation', 'relu')
            )

        elif encoder.lower() in ULMFIT_CLASSIFIER:
            lm_pretrain = config[MODEL][INPUT_FEATURES].get('pretrain_language_model', None)

            # Fine-tuning LM from Training Dataset
            lm_trainer = LanguageModelTrainer(pretrain=lm_pretrain)
            lm_trainer.fine_tuning_from_df(data_df=data_source.train.data,
                                        batch_size=training_params.get('batch_size', 128))

            # define model
            learn = ULMFITClassificationLearner(
                mode=TRAINING_MODE, 
                data_source=data_source, 
                drop_mult=encoder_args.get('drop_mult', 0.3), 
                average=encoder_args.get('average', 'weighted'), 
                beta=encoder_args.get('beta', 1)
            )

        elif encoder.lower() in FLAIR_SEQUENCE_TAGGER:
            embedding_types = config[MODEL][INPUT_FEATURES].get('embedding_types', None)
            pretrain_embedding = config[MODEL][INPUT_FEATURES].get('pretrain_embedding', None)

            embeddings = Embeddings(
                embedding_types=embedding_types, pretrain=pretrain_embedding)
            embedding = embeddings.embed()

            learn = FlairSequenceTaggerLearner(
                mode=TRAINING_MODE,
                data_source=data_source,
                tag_type=_type,
                embeddings=embedding,
                hidden_size=encoder_args.get('hidden_size', 1024),
                use_rnn=encoder_args.get('use_rnn', True),
                rnn_layers=encoder_args.get('rnn_layers', 1),
                dropout=encoder_args.get('dropout', 0.0),
                word_dropout=encoder_args.get('word_dropout', 0.05),
                locked_dropout=encoder_args.get('locked_dropout', 0.5),
                reproject_embeddings=encoder_args.get('reproject_embeddings', True),
                use_crf=decoder.get('crf', True),
                beta=encoder_args.get('beta', 1)
            )
                                            
        else:
            raise ValueError(
                f'This `{encoder}` encoder model is not supported.')

        # Get params training model
        batch_size = training_params.get('batch_size')
        learning_rate = float(training_params.get('learning_rate'))
        moms = training_params.get('momentums')
        
        if early_stopping_epochs:
            num_epochs = early_stopping_epochs
        else:
            num_epochs = training_params.get('num_epochs')

        trainer = Trainer(learn=learn)
        results = trainer.train(
                            base_path=base_path,
                            learning_rate=learning_rate,
                            batch_size=batch_size,
                            num_epochs=num_epochs,
                            monitor_test=True,
                            verbose=verbose, 
                            model_file=model_file,
                            moms=moms, 
                            # verbose=False, 
                            skip_save_model=skip_save_model, 
                            is_save_best_model=is_save_best_model
                        )
        return results


    @classmethod
    def get_predict(
        cls, 
        config: Union[str, dict, Path], 
        data_path: Union[str, Path]
    ):
        if isinstance(config, str) or isinstance(config, Path):
            config = get_config_yaml(config_file=config)

        _type = config[MODEL][OUTPUT_FEATURES][TYPE]

        data_path = ifnone(data_path, config[DATASET].get(TEST_PATH))

        pre_processing = config[DATASET][PRE_PROCESSING]

        encoder = config[MODEL][INPUT_FEATURES][ENCODER][NAME]

        base_path = config[TRAINING_PARAMS].get('base_path')
        model_file = config[TRAINING_PARAMS].get('model_file')
        model_path = Path(base_path)/model_file

        text_col = config[DATASET].get(TEXT_COL, 'text')
        label_col = config[DATASET].get(LABEL_COL, None)
        intent_col = config[DATASET].get(INTENT_COL, None)
        tag_col = config[DATASET].get(TAG_COL, None)

        is_lowercase = pre_processing.get(LOWERCASE_TOKEN, True)
        rm_emoji = pre_processing.get(REMOVE_EMOJI, False)
        rm_special_token = pre_processing.get(REMOVE_SPECIAL_TOKEN, False)
        rm_url = pre_processing.get(REMOVE_URL, False)

        if encoder.lower() in ULMFIT_CLASSIFIER:
            learn = ULMFITClassificationLearner(mode=INFERENCE_MODE, model_path=model_path)
        elif encoder.lower() in FLAIR_SEQUENCE_TAGGER:
            learn = FlairSequenceTaggerLearner(mode=INFERENCE_MODE, model_path=model_path)
        elif encoder.lower() in ONENET_NLU:
            learn = OnenetLearner(mode=INFERENCE_MODE, model_path=model_path)
        else:
            raise ValueError(
                f'This `{encoder}` encoder model is not supported.')

        data_df = learn.predict_on_df(
                            data=data_path, 
                            text_col=text_col,
                            label_col=label_col,
                            intent_col=intent_col,
                            tag_col=tag_col,
                            lowercase=is_lowercase, 
                            rm_emoji=rm_emoji, 
                            rm_url=rm_url, 
                            rm_special_token=rm_special_token
                        )
                                            
        return data_df
