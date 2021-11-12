# -*- coding: utf-8
# Copyright (c) 2021 by Phuc Phan

import os
import logging
import numpy as np

from tqdm import tqdm
from typing import Union
from pathlib import Path
from pandas import DataFrame
from fastai.text import DatasetType
from sklearn.metrics import confusion_matrix
from fastai.text import (
    text_classifier_learner, accuracy, FBeta, Precision, Recall,
    load_learner, AWD_LSTM )

from denver.constants import *
from denver.data import normalize
from denver.metrics import get_metric
from denver.data import DenverDataSource
from denver.learners import DenverLearner
from denver import MODE_SUPPORTED, DENVER_DIR
from denver.utils.utils import rename_file, ifnone
from denver.utils.print_utils import plot_confusion_matrix
from denver.uncertainty_estimate import UncertaintyEstimator

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class ULMFITClassificationLearner(DenverLearner):
    """A class ULMFIT Classification Learner. """
    def __init__(
        self,
        data_source: DenverDataSource=None,
        mode: str="training",
        model_path: Union[str, Path]=None,
        drop_mult: float=0.3, 
        average: str='weighted', 
        beta: int=1,
    ):  
        """
        Initialize a ULMFITClassifer 
        
        :param data_source: A DenverDataSource class
        :param device: The device to use
        :param mode: The mode {inference, training, tuning}
        :param model_path: The path to model with mode inference
        :param drop_mult: The dropout multiple
        :param average: The average in 'binary', 'micro', 'macro', 'weighted' or None
        :param beta: Parameter for F-beta score
        """
        super(ULMFITClassificationLearner, self).__init__()

        self.mode = mode
        self.data_source = data_source

         # Verify mode option
        if mode not in MODE_SUPPORTED:
            raise ValueError(
                f"Not support mode: `{mode}``. "
                f"Please select 1 among these support modes: {''.join(MODE_SUPPORTED)}")
        
        if self.mode == INFERENCE_MODE:
            if model_path is not None:
                model_path = os.path.abspath(model_path)
            else:
                logger.error(f"MODE: `inference`: `model_path` is None value.")
            if os.path.exists(model_path):
                logger.debug(f"Load the model from path: {model_path}")
                self.model = self._load_model(model_path)
            else:
                raise ValueError(f"`{model_path}` not supplied or not found.")

        else:
            if data_source is not None:
                self.vocab, self.data_clas = self.data_source.build_databunch()
            else:
                raise ValueError(
                    f"With mode in ['training', 'tunning'] need to pass into `DenverDataSource`. ")
                    
            self.model = text_classifier_learner(self.data_clas, AWD_LSTM, drop_mult=drop_mult,
                                                 metrics=[accuracy,
                                                          FBeta(
                                                              average=average, beta=beta),
                                                          Precision(
                                                              average=average),
                                                          Recall(average=average)])
    
    @property
    def __name__(self):
        return 'ULMFITClassifier'

    def _load_model(self, model_path):
        '''Load a saved model with the model path

        :param model_path: path to the model

        :returns: The loaded model.
        '''
        return load_learner('/', model_path)
    
    def _load_encoder(self, pretrained_path):
        '''Load a saved weight with the pretrained path.

        :param pretrained_path: path to the pretrained model
        '''
        self.model.load_encoder(pretrained_path)

    def get_classes(self):
        """Get a model class list

        :returns: List classes from model.
        """
        return self.model.data.classes

    def unfreeze(self):
        '''
        Unfreeze entire model.
        '''
        self.model.unfreeze()

    def freeze(self):
        '''
        Freeze up to last layer group.
        '''
        self.model.freeze()

    def freeze_to(self, num_layers:int):
        '''
        Freeze layers up to layer group `num_layers`
        '''
        self.model.freeze_to(num_layers)

    def fit(self, learning_rate=1e-3, moms=(0.7, 0.8), num_epochs: int=5):
        '''Fit the model on data and learn using learning rate, momentums and number of epochs.

        :param learning_rate: learning rate
        :param moms: A dict momentums rate, use when train model IC with fastai
        :param num_epochs: The number of epochs
        '''
        self.model.fit_one_cycle(num_epochs, learning_rate, moms)

    def train(
        self, 
        base_path: Union[str, Path]='./models/', 
        is_save_best_model: bool=True,
        model_file: str=None, 
        learning_rate: float=2e-2, 
        batch_size: int=128, 
        num_epochs: int=15, 
        skip_save_model: bool=False, 
        clean_lm: bool=True,
        **kwargs
    ):
        '''Training the model

        :param base_path: Main path to models are saved
        :param is_save_best_model: If True save the best model, else save the final model
        :param model_file: The file name to save the model
        :param learning_rate: learning rate
        :param batch_size: batch size
        :param num_epochs: The number of epochs 
        :param skip_save_model: If True, disables saving the model
        '''

        moms = kwargs.get('moms')
        moms = ifnone(moms, [0.8, 0.7])

        pretrained_path = os.path.abspath(f'{DENVER_DIR}/vifine_tuned_enc_ic')
        if os.path.exists(pretrained_path + '.pth'):
            self.model.load_encoder(pretrained_path)
        else:
            raise ValueError(f"Encode pretrain language model `{pretrained_path}` not found. "
                            f"You must fine-tuning language model from training dataset. "
                            f"See detail in the function `fine_tuning_from_df` in `language_model_trainer`.")

        learning_rate *= batch_size / 48

        ## 1
        self.model.freeze()
        self.model.fit_one_cycle(3, learning_rate, moms=moms)
        ## 2
        self.model.fit_one_cycle(3, learning_rate, moms=moms)
        ## 3
        self.model.freeze_to(-2)
        self.model.fit_one_cycle(1, slice(learning_rate / (2.6 ** 4), learning_rate), moms=moms)
        ## 4
        self.model.freeze_to(-3)
        self.model.fit_one_cycle(1, slice(learning_rate / 2 / (2.6 ** 4), learning_rate / 2), moms=moms)
        ## 5
        self.model.unfreeze()
        self.model.fit_one_cycle((num_epochs-8), slice(learning_rate / 10 / (2.6 ** 4), learning_rate / 10), moms=(0.8, 0.7))
        
        
        if not skip_save_model:
            logger.info(f"Save the model...")
            model_file = model_file if model_file else "denver.pkl"
            self.save(base_path=base_path, save_file=model_file)
        
        self.valid_results = self.model.validate()

        if clean_lm:
            try:
                os.remove(os.path.abspath(f"{DENVER_DIR}/vifine_tuned_enc_ic.pth"))
                os.remove(os.path.abspath(f"{DENVER_DIR}/vifine_tuned_ic.pth"))
            except Exception as e:
                logger.warning(f"{e}")

    def validate(self):
        """
        :returns: The calculated loss and the metrics of the current model on the given valid data. 
        """
        metrics = {
            'loss': [self.valid_results[0]],
            'acc': [self.valid_results[1]], 
            'f1': [self.valid_results[2]], 
            'precision': [self.valid_results[3]], 
            'recall': [self.valid_results[4]]
            }
        return metrics

    def evaluate_by_step(
        self, 
        data: Union[str, DataFrame, DenverDataSource]=None, 
        text_col: str='text',
        label_col: str='label', 
        lowercase: bool=True, 
        rm_emoji: bool=True, 
        rm_url: bool=True, 
        rm_special_token: bool=False, 
        with_confusion_matrix: bool=False, 
        save_dir: str='./evaluation/', 
        save_name: str='confusion-matrix'
    ):
        """Evalute the model and storage confustion matrics chart
            
        :param data: The path to data (.csv) or a DataFrame or a DenverDataSource
        :param text_col: The column name of text data
        :param label_col: The column name of label data
        :param lowercase: If True, lowercase data
        :param rm_emoji: If True, replace the emoji token into <space> (" ")
        :param rm_url: If True, replace the url token into <space> (" ")
        :param rm_special_token: If True, replace the special token into <space> (" "),
                                 special token included of punctuation token, characters without vietnamese characters
        :param with_confusion_matrix: If True, save confusion matrix figure
        :param save_cm_dir: The path directory to save confusion-matrix's image
        :param save_cm_dir: The file name to save confusion-matrix's image

        :retuns: Metrics format:\n
                    {\n
                        'acc': acc, \n
                        'f1': f1, \n
                        'precision': precision, \n
                        'recall': recall\n
                    }\n
        """

        logger.info(f"Evaluating...")
        classes = self.get_classes()

        data_source = data
        if isinstance(data, str):
            if not data.endswith('.csv'):
                raise ValueError(
                    f"File `{data}` is invalid file format .csv.")
            data_path = os.path.abspath(data)
            data_source = DenverDataSource.from_csv(test_path=data_path, 
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
        
        inputs = data_source.test.get_sentences()
        labels = data_source.test.get_labels()

        preds = []

        for i in tqdm(range(len(inputs))):
            id_pred = self.model.predict(inputs[i])[1]
            preds.append(classes[id_pred])
        
        acc, f1, precision, recall, detailed_results = get_metric(labels, preds)

        metrics = {
            'acc': [acc.item()], 
            'f1': [f1.item()], 
            'precision': [precision.item()], 
            'recall': [recall.item()], 
            'cls_detailed_results': detailed_results,
            }

        if with_confusion_matrix:
            # Plot confusion matrix
            cm = confusion_matrix(y_true=labels, y_pred=preds, labels=classes)

            save_dir = os.path.abspath(save_dir)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            plot_confusion_matrix(cm, normalize=True, target_names=classes, title=save_name + '1', save_dir=save_dir)
            plot_confusion_matrix(cm, normalize=False, target_names=classes, title=save_name + '2', save_dir=save_dir)
            logger.info(f"Path to saved confusion matrix: {save_dir}/{save_name + 'x'}.png")

        return metrics

    def evaluate(
        self, 
        data: Union[str, DataFrame, DenverDataSource]=None, 
        text_col: str='text',
        label_col: str='label', 
        lowercase: bool=True, 
        rm_emoji: bool=True, 
        rm_url: bool=True, 
        rm_special_token: bool=False, 
        with_confusion_matrix: bool=False, 
        save_dir: str='./evaluation/', 
        save_name: str='confusion-matrix', 
        **kwargs
    ):
        """Batch evalute the model and storage confustion matrics chart
            
        :param data: The path to data or a DataFrame
        :param text_col: The column name of text data
        :param label_col: The column name of label data
        :param lowercase: If True, lowercase data
        :param rm_emoji: If True, replace the emoji token into <space> (" ")
        :param rm_url: If True, replace the url token into <space> (" ")
        :param rm_special_token: If True, replace the special token into <space> (" "),
                                 special token included of punctuation token, characters without vietnamese characters
        :param confusion_matrix: If True, save confusion matrix figure
        :param save_cm_dir: The path directory to save confusion-matrix's image
        :param save_cm_dir: The file name to save confusion-matrix's image

        :retuns: Metrics format:\n
                    {\n
                        'acc': acc, \n
                        'f1': f1, \n
                        'precision': precision, \n
                        'recall': recall\n
                    }\n
        """

        logger.info(f"Evaluating...")
        classes = self.get_classes()
        logger.debug(f"Classes: {len(classes)} - {classes}")

        data_source = data
        if isinstance(data, str):
            if not data.endswith('.csv'):
                raise ValueError(
                    f"File `{data}` is invalid file format .csv.")
            data_path = os.path.abspath(data)
            data_source = DenverDataSource.from_csv(test_path=data_path, 
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
        
        self.model.data.add_test(data_source.test.data)
        prob_preds, _ = self.model.get_preds(ds_type=DatasetType.Test, ordered=True)
        
        preds = [classes[i] for i in prob_preds.argmax(1)]
        labels = data_source.test.get_labels()

        acc, f1, precision, recall, detailed_results = get_metric(labels, preds)

        metrics = {
            'acc': [acc.item()], 
            'f1': [f1.item()], 
            'precision': [precision.item()], 
            'recall': [recall.item()], 
            'cls_detailed_results': detailed_results,
            }
        if with_confusion_matrix:
            # Plot confusion matrix
            cm = confusion_matrix(y_true=labels, y_pred=preds, labels=classes)

            save_dir = os.path.abspath(save_dir)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            plot_confusion_matrix(cm, normalize=True, target_names=classes, title=save_name + '1', save_dir=save_dir)
            plot_confusion_matrix(cm, normalize=False, target_names=classes, title=save_name + '2', save_dir=save_dir)
            logger.info(f"Path to saved confusion matrix: {save_dir}/{save_name + 'x'}.png")

        return metrics


    def predict(
        self, 
        sample:str, 
        with_dropout:bool=False, 
        lowercase: bool=True, 
        rm_emoji: bool=True, 
        rm_url: bool=True, 
        rm_special_token: bool=False, 
    ):
        """Predicts the class labels for the given sample
        
        :param sample: the sample need to predict
        :param with_dropout: if True, use mc dropout during the predict
        :param lowercase: If True, lowercase data
        :param rm_emoji: If True, replace the emoji token into <space> (" ")
        :param rm_url: If True, replace the url token into <space> (" ")
        :param rm_special_token: If True, replace the special token into <space> (" "),
                                 special token included of punctuation token, characters without vietnamese characters

        :returns: Return predicted class, label and probabilities.
        """
        sample = normalize(sample, rm_emoji=rm_emoji, rm_url=rm_url, 
                        rm_special_token=rm_special_token, lowercase=lowercase)

        return self.model.predict(sample, with_dropout=with_dropout)

    def predict_with_mc_dropout(
        self, 
        sample, 
        lowercase: bool=True, 
        rm_emoji: bool=True, 
        rm_url: bool=True, 
        rm_special_token: bool=False, 
        n_times=10
    ):
        """Predicts the class labels for the given sampels with MC Dropout.
        
        :param sample: the sample need to predict
        :param lowercase: If True, lowercase data
        :param rm_emoji: If True, replace the emoji token into <space> (" ")
        :param rm_url: If True, replace the url token into <space> (" ")
        :param rm_special_token: If True, replace the special token into <space> (" "), 
                                 special token included of punctuation token, characters without vietnamese characters
        :param n_times: the times prediction to calculate with MC Dropout.

        :returns: Return List predicted class, label and probabilities of `n_times` times.
        """
        sample = normalize(sample, rm_emoji=rm_emoji, rm_url=rm_url, 
                        rm_special_token=rm_special_token, lowercase=lowercase)

        return self.model.predict_with_mc_dropout(sample, n_times=n_times)

    def get_uncertainty_score(
        self, 
        sample, 
        lowercase: bool=True, 
        rm_emoji: bool=True, 
        rm_url: bool=True, 
        rm_special_token: bool=False,  
        n_times=10
    ):
        """Get uncertainty score of the prediction of a sample.

        :param sample: The sample need to calculate uncertainty score.
        :param lowercase: If True, lowercase data
        :param rm_emoji: If True, replace the emoji token into <space> (" ")
        :param rm_url: If True, replace the url token into <space> (" ")
        :param rm_special_token: If True, replace the special token into <space> (" "),
                                 special token included of punctuation token, characters without vietnamese characters
        :param n_times: The times prediction to calculate uncertainty score.
        
        :returns: Return the value format:\n
                    {\n
                        'text': sample,\n
                        'intent': iclass,\n
                        'uncertainty_score': value,\n
                        'method': method.__name__\n
                    }\n
        """
        verbose = True
        sample = normalize(sample, rm_emoji=rm_emoji, rm_url=rm_url, 
                        rm_special_token=rm_special_token, lowercase=lowercase)

        uncertainty_estimator = UncertaintyEstimator(verbose)
        uncertainty_score = uncertainty_estimator.get_uncertainty_score(uncertainty_estimator.entropy, 
                                                                        self.model, sample=sample, n_times=n_times)
        return uncertainty_score
    
    def convert_to_rasa_format(self, classes, output):
        """
        Convert output in rasa's format
        
        :param classes: List of model classes
        :param output: Output in fastai format

        :returns: A result rasa format.
        """

        output = output.cpu().detach().numpy().astype(float)
        idx_max = np.argmax(output)

        intent = {"name": classes[idx_max], "confidence": output[idx_max].item()}

        intent_ranking = [
            {"name": classes[score], "confidence": output[score].item()}
            for score in range(len(output))
        ]

        output = {
            "intent": intent,
            "intent_ranking": intent_ranking
        }

        return output

    def process(
        self, 
        sample, 
        lowercase: bool=True, 
        rm_emoji: bool=True, 
        rm_url: bool=True, 
        rm_special_token: bool=False
    ):
        '''Return the results as output of rasa format

        :param sample: The sample need to calculate uncertainty score.
        :param lowercase: If True, lowercase data
        :param rm_emoji: If True, replace the emoji token into <space> (" ")
        :param rm_url: If True, replace the url token into <space> (" ")
        :param rm_special_token: If True, replace the special token into <space> (" "), 
                                 special token included of punctuation token, characters without vietnamese characters

        :returns: The result format rasa
        '''
        sample = normalize(sample, rm_emoji=rm_emoji, rm_url=rm_url, 
                        rm_special_token=rm_special_token, lowercase=lowercase)

        rasa_output = self.convert_to_rasa_format(self.get_classes(), self.model.predict(sample)[2])
        return rasa_output

    def predict_on_df_by_step(
        self, 
        data: Union[str, Path, DataFrame], 
        text_col: str = 'text',
        label_col: str = None,
        with_dropout: bool=False,
        with_uncertainty: bool=False,
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
        :param with_dropout: if True, use mc dropout during the predict.
        :param with_uncertainty_score: if True, consist of uncertainty score
        :param lowercase: If True, lowercase data
        :param rm_emoji: If True, replace the emoji token into <space> (" ")
        :param rm_url: If True, replace the url token into <space> (" ")
        :param rm_special_token: If True, replace the special token into <space> (" "),
                                 special token included of punctuation token, characters without vietnamese characters

        :returns: A DataFrame include of columns {'text', 'pred', 'uncertainty_score'}.
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
        
        preds = []
        uncertainty_scores = []

        logger.info(f"Get-prediction...")
        for i in tqdm(range(len(data_df))):
            output = self.predict(
                sample=data_df['text'][i], 
                with_dropout=with_dropout, 
                rm_emoji=rm_emoji, 
                rm_url=rm_url, 
                rm_special_token=rm_special_token, 
                lowercase=lowercase
            )

            preds.append(output[0].obj)

            if with_uncertainty:
                uncertainty_score = self.get_uncertainty_score(
                                            sample=data_df['text'][i], 
                                            rm_emoji=rm_emoji, 
                                            rm_url=rm_url, 
                                            rm_special_token=rm_special_token, 
                                            lowercase=lowercase
                                        )

                uncertainty_score = uncertainty_score['uncertainty_score']
                uncertainty_scores.append(uncertainty_score)
        
        data_df['pred'] = preds
        if with_uncertainty:
            data_df['uncertainty_score'] = uncertainty_scores

        return data_df

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
        """Batch predicts the class labels for a DataFrame or a path to .csv file
        
        :param data: The path to data or a DataFrame or a path to .csv file
        :param text_col: The column name of text data
        :param label_col: The column name of label data
        :param lowercase: If True, lowercase data
        :param rm_emoji: If True, replace the emoji token into <space> (" ")
        :param rm_url: If True, replace the url token into <space> (" ")
        :param rm_special_token: If True, replace the special token into <space> (" "), 
                                 special token included of punctuation token, characters without vietnamese characters

        :returns: A DataFrame include of columns {'text', 'pred', 'uncertainty_score'}.
        """
        classes = self.get_classes()
        logger.debug(f"Classes: {len(classes)} - {classes}")

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
        
        preds = []

        logger.info(f"Get-prediction...")
        self.model.data.add_test(data_df)
        prob_preds, _ = self.model.get_preds(ds_type=DatasetType.Test, ordered=True)
        
        preds = [classes[i] for i in prob_preds.argmax(1)]
        
        data_df['pred'] = preds

        data_df = data_df.sort_values(by='pred')

        return data_df
        
    def save(
        self, 
        base_path: Union[str, Path], 
        save_file: Union[str, Path], 
        overwrite: bool=True
    ):
        """Export the model

        :param save_file: Save the model to the save_file 
        :param overwrite: If True, overwrite the old file.
        """
        file_path = os.path.abspath(Path(base_path)/save_file)
        if not overwrite:
            rename_file(file_path)

        try:
            self.model.export(file_path)
            logger.info(f"Path to the saved model: {file_path}")
        except:
            logger.error(f"Can't save model into: {file_path}")
