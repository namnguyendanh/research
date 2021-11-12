# -*- coding: utf-8
# Copyright (c) 2021 by Phuc Phan

import os
import denver

from tqdm import tqdm
from typing import Union
from pathlib import Path
from pandas import DataFrame
from flair.data import Sentence
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger

from denver.constants import *
from denver import MODE_SUPPORTED
from denver.data import normalize
from denver.data import DenverDataSource
from denver.learners import DenverLearner
from denver.utils.utils import convert_to_BIO
from denver.embeddings.embeddings import Embeddings

import logging
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class FlairSequenceTaggerLearner(DenverLearner):
    """A class Flair Sequence Tagger Learner. """
    def __init__(
        self, 
        data_source: DenverDataSource=None, 
        tag_type: str='ner', 
        embeddings: Embeddings=None, 
        mode: str="training", 
        model_path: str=None, 
        hidden_size: int=1024, 
        use_rnn: bool=True,
        rnn_layers: int=1, 
        dropout: float=0.0, 
        word_dropout: float=0.05, 
        locked_dropout: float=0.5, 
        reproject_embeddings: Union[bool,int] = True,
        use_crf: bool=True, 
        beta: float=1,
    ):
        """Initialize a FlairSequenceTagger Learner

        :param data_source: A DenverDataSource class
        :param tag_type: The type of tagging
        :param embeddings: The embeddings that you want to train. 
                           The embeddings should be from denver.embeddings.embeddings.Embedding
        :param mode: Mode in {inference, training}
        :param model_path: The path to model, (mode = inference)
        :param hidden_size: The number of hidden states in RNN
        :param use_rnn: If True use RNN layer, otherwise use word embeddings directly
        :param rnn_layers: The number of RNN layers
        :param dropout: dropout probability
        :param word_dropout: word dropout probability
        :param locked_dropout: locked dropout probability
        :param reproject_embeddings: if True, adds trainable linear map on top of embedding layer. If False, no map.
        :param use_crf: If True use CRF decoder, else project directly to tag space
        :param beta: Parameter for F-beta score for evaluation and training annealing
        """
        super(FlairSequenceTaggerLearner, self).__init__()
        
        # Verify mode option
        if mode not in MODE_SUPPORTED:
            raise ValueError(
                f"\nNot support mode: {mode}."
                f"Please select 1 among these support modes: {''.join(MODE_SUPPORTED)}")

        self.mode = mode
        self.data_source = data_source
        self.tag_type = tag_type
        self.embeddings = embeddings
        self.hidden_size = hidden_size
        self.use_rnn = use_rnn
        self.rnn_layers = rnn_layers
        self.dropout = dropout
        self.word_dropout = word_dropout
        self.locked_dropout = locked_dropout
        self.use_crf = use_crf
        self.reproject_embeddings = reproject_embeddings
        self.beta = beta

        if self.data_source is not None:
            self.data_corpus = self.data_source.build_corpus()
        else:
            self.data_corpus = None

        if self.mode == INFERENCE_MODE:
            if model_path is not None:
                self.model_path = os.path.abspath(model_path)
            else:
                logger.error(f"  MODE `inference` need to `model_path` not None.")

            if os.path.exists(self.model_path):
                logger.debug(
                    f"  Load the model from path: {self.model_path}")
                self.model = self._load_model(self.model_path)
            else:
                logger.debug(
                    f"  The model not supplied or not found." 
                    f"Download model the default model (vi_ner.pt) from out server.")

    @property
    def __name__(self):
        return 'FlairSequenceTagger'
                        
    def _load_model(self, model_path):
        model = SequenceTagger.load(self.model_path)
        return model
        
    def train(
        self, 
        base_path: Union[str, Path]='./models/', 
        is_save_best_model: bool=True,
        model_file: str=None, 
        learning_rate: float=0.1, 
        batch_size: int=32, 
        num_epochs: int=500, 
        skip_save_model: bool=False, 
        **kwargs
    ):
        """Training the learner

        :param base_path: Main path to models are saved
        :param is_save_best_model: If True save the best model, else save the final model
        :param model_file: The file name to save the model
        :param learning_rate: Initial learning rate
        :param batch_size: Size of batches during training
        :param num_epochs: The number of epochs to train
        :param skip_save_model: If True, disables saving the model
        """
        base_path = os.path.abspath(base_path)

        tag_dictionary = self.data_corpus.make_tag_dictionary(tag_type=self.tag_type)

        tagger: SequenceTagger = SequenceTagger(
            hidden_size=self.hidden_size,
            tag_type=self.tag_type,
            embeddings=self.embeddings,
            tag_dictionary=tag_dictionary,
            use_rnn=self.use_rnn,
            rnn_layers=self.rnn_layers,
            dropout=self.dropout,
            word_dropout=self.word_dropout,
            locked_dropout=self.locked_dropout,
            reproject_embeddings=self.reproject_embeddings,
            beta=self.beta,
            use_crf=self.use_crf,
        )

        # Initialize trainer
        trainer: ModelTrainer = ModelTrainer(tagger, self.data_corpus)
        save_final_model = False if is_save_best_model else True
        trainer.train(
            base_path, 
            learning_rate=learning_rate,
            mini_batch_size=batch_size,
            # train_with_dev=train_with_dev,
            embeddings_storage_mode=denver.device,
            anneal_with_restarts=is_save_best_model,
            save_final_model=save_final_model, 
            skip_save_model=skip_save_model, 
            max_epochs=num_epochs,
        )
        self.model = tagger

        if is_save_best_model and not skip_save_model:
            if model_file:
                os.rename(os.path.abspath(Path(base_path)/'best-model.pt'), 
                          os.path.abspath(Path(base_path)/model_file))  
                logger.info(f"Path to the saved model: {Path(base_path)/model_file}")
            else:
                logger.info(f"Path to the saved model: {Path(base_path)/'best-model.pt'}")
        elif not skip_save_model:
            if model_file:
                os.rename(os.path.abspath(Path(base_path)/'final-model.pt'), 
                          os.path.abspath(Path(base_path)/model_file))  
                logger.info(f"Path to the saved model: {Path(base_path)/model_file}")
            else:
                logger.info(f"Path to the saved model: {Path(base_path)/'final-model.pt'}")

    def predict(
        self, 
        sample:str, 
        lowercase: bool=True, 
        rm_emoji: bool=True, 
        rm_url: bool=True, 
        rm_special_token: bool=False
    ):
        """Predicts the tags for the given sequence sample

        :param sample: The sample need to predict
        :param lowercase: If True, lowercase data
        :param rm_emoji: If True, replace the emoji token into <space> (" ")
        :param rm_url: If True, replace the url token into <space> (" ")
        :param rm_special_token: If True, replace the special token into <space> (" "), 
                                 special token included of punctuation token, characters without vietnamese characters

        :returns: Return the predicted tagging
        """
        sample = normalize(sample, rm_emoji=rm_emoji, rm_url=rm_url, 
                        rm_special_token=rm_special_token, lowercase=lowercase)

        sample = Sentence(sample)
        self.model.predict(sample)

        return sample.to_tagged_string()
    
    def predict_on_df(
        self, 
        data: Union[str, Path, DataFrame], 
        text_col: str = 'text',
        label_col: str = None,
        lowercase: bool=True, 
        rm_emoji: bool=True, 
        rm_url: bool=True, 
        rm_special_token: bool=False,
        **kwargs
    ):
        """Predicts the class labels for a DataFrame or a path to .csv file
        
        :param data: The path to data or a DataFrame or a path to .csv file
        :param text_col: The column name of text data
        :param label_col: The column name of label data
        :param lowercase: If True, lowercase data
        :param rm_emoji: If True, replace the emoji token into <space> (" ")
        :param rm_url: If True, replace the url token into <space> (" ")
        :param rm_special_token: If True, replace the special token into <space> (" "), 
                                 special token included of punctuation token, characters without vietnamese characters

        :returns: A DataFrame include of columns {'text', 'pred','label'(maybe)}.
        """

        if isinstance(data, str) or isinstance(data, Path):
            data_source = DenverDataSource.from_csv(test_path=data,
                                                    text_col=text_col, 
                                                    label_col=label_col,
                                                    rm_emoji=rm_emoji, 
                                                    rm_url=rm_url, 
                                                    rm_special_token=rm_special_token, 
                                                    lowercase=lowercase)
            data_df = data_source.test.data
        else:
            data_df = data
        
        if 'text' not in data_df.columns:
            raise ValueError(f"Column name in DataFrame must be include `text` column.")
        
        preds = []

        logger.info(f"Get-prediction...")
        for i in tqdm(range(len(data_df))):
            output = self.process(
                            sample=data_df['text'][i], 
                            rm_emoji=rm_emoji, 
                            rm_url=rm_url, 
                            rm_special_token=rm_special_token, 
                            lowercase=lowercase
                        )

            bio = convert_to_BIO(entities=output, text=data_df['text'][i])
            preds.append(bio)
        
        data_df['pred'] = preds

        return data_df

    def validate(self):
        """Evaluate the model with dev dataset

        :returns: dev_eval_result: The result.
        :returns: dev_loss: The value loss.
        """
        if self.data_corpus.dev is not None:
            result, loss = self.model.evaluate(self.data_corpus.dev)
            metrics = {
                'loss': [loss],
                'main_score': [result.main_score],
                'detailed_results': [result.detailed_results],
            }
            return metrics
        else:
            logger.debug(f"`dataset` is None value.")
            return None

    def evaluate(
        self, 
        data: Union[str, DataFrame, DenverDataSource]=None, 
        text_col: str='text',
        label_col: str='label',
        lowercase: bool=True, 
        rm_emoji: bool=True, 
        rm_url: bool=True, 
        rm_special_token: bool=False, 
        **kwargs
    ):
        """Evaluate the model with test dataset

        :param data: The path of eval dataset (.csv) or a DataFrame or a DenverDataSource
        :param text_col: The column name of text data
        :param label_col: The column name of label data
        :param lowercase: If True, lowercase data
        :param rm_emoji: If True, replace the emoji token into <space> (" ")
        :param rm_url: If True, replace the url token into <space> (" ")
        :param rm_special_token: If True, replace the special token into <space> (" "), 
                                 special token included of punctuation token, characters without vietnamese characters

        :returns: Metrics \n
                    {\n
                        'loss': loss, \n
                        'main_score': main_score, \n 
                        'detailed_results': detailed_results, \n
                    } \n
        """
        data_source = data
        if isinstance(data, str):
            if not data.endswith('.csv'):
                raise ValueError(
                    f"File `{data}` is invalid file format .csv.")
            data = os.path.abspath(data)
            data_source = DenverDataSource.from_csv(test_path=data,
                                                    text_col=text_col,
                                                    label_col=label_col,
                                                    rm_emoji=rm_emoji, 
                                                    rm_url=rm_url, 
                                                    rm_special_token=rm_special_token, 
                                                    lowercase=lowercase)
        elif isinstance(data, DataFrame):
            data_source = DenverDataSource.from_df(test_df=data,
                                                   text_col=text_col,
                                                   label_col=label_col,
                                                   rm_emoji=rm_emoji, 
                                                    rm_url=rm_url, 
                                                    rm_special_token=rm_special_token, 
                                                    lowercase=lowercase)

        elif (data is None) and (self.data_source.test is not None):
            data_source = self.data_source
        elif not isinstance(data, DenverDataSource):
            raise ValueError(f"`data` must be difference to `None` value.")

        corpus = data_source.build_corpus()
        logger.info(f"Evaluating...")
        result, loss = self.model.evaluate(corpus.test)

        metrics = {
            'loss': [loss],
            'main_score': [result.main_score],
            'tag_detailed_results': [result.detailed_results],
        }

        return metrics

    def convert_to_rasa_format(self, entities):
        """
        Convert output in rasa's format
        
        :param entities: List entities from the output of process function.

        :returns: A result rasa format. Example: \n
                    [{\n
                        'confidence': confidence,\n
                        'end': end,\n
                        'start': start,\n
                        'entity': entity,\n
                        'extractor': extractor,\n
                        'value': value  \n
                    }]\n
        """
        
        entities_rasa = []

        for entity in entities:
            label = entity['labels'][0].value
            confidence = entity['labels'][0].score

            if 'attribute' in label:
                value = label[label.index(':') + 1:]
                entity_name = label[:label.index(':')]
            else:
                value = entity['text']
                entity_name = label

            entity_rasa = {
                "start": entity['start_pos'],
                "end": entity['end_pos'],
                "value": value,
                "entity": entity_name,
                "confidence": confidence,
                "extractor": self.__name__
            }
            entities_rasa.append(entity_rasa)

        return entities_rasa

    def process(
        self, 
        sample, 
        lowercase: bool=True, 
        rm_emoji: bool=True, 
        rm_url: bool=True, 
        rm_special_token: bool=False
    ):
        """Return the results as output of rasa format

        :param sample: The sample need to calculate uncertainty score.
        :param lowercase: If True, lowercase data
        :param rm_emoji: If True, replace the emoji token into <space> (" ")
        :param rm_url: If True, replace the url token into <space> (" ")
        :param rm_special_token: If True, replace the special token into <space> (" "),
                                 special token included of punctuation token, characters without vietnamese characters

        :returns: The results format rasa. Example:
                    [{\n
                        'confidence': 0.9994359612464905,\n
                        'end': 19,\n
                        'entity': 'object_type',\n
                        'extractor': 'FlairSequenceTagger',\n
                        'start': 16,\n
                        'value': 'cũi'\n
                    }]\n
        """
        sample = normalize(sample, rm_emoji=rm_emoji, rm_url=rm_url, 
                    rm_special_token=rm_special_token, lowercase=lowercase)

        sample = Sentence(sample)
        self.model.predict(sample)
        rasa_output = self.convert_to_rasa_format(sample.to_dict(tag_type=self.tag_type)['entities'])
        return rasa_output
