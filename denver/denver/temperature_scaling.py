# -*- coding: utf-8
# Copyright (c) 2021 by Phuc Phan

import os
import torch
import logging
import numpy as np

from tqdm import tqdm
from torch import nn, optim
from pandas import DataFrame
from typing import Text, Union
from scipy.special import softmax

from denver.functional import ECELoss
from denver.data import DenverDataSource
from denver.learners import DenverLearner
from denver.utils.print_utils import ConfidenceHistogram, ReliabilityDiagram

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class ModelWithTeperature(nn.Module):
    def __init__(self, learner: DenverLearner=None, device: str=None):
        """
        A thin decorate, which wraps a model with temperature scaling model.

        :pararm learner (nn.Module): A DenverLearner. The output of `process` function` should be have `intent_logits`,
                                   NOT the softmax (or log softmax)!
        """
        super(ModelWithTeperature, self).__init__()
        
        self.learner = learner
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        if self.learner:
            self.intent_classes = self.learner.get_classes().get('intent', {})
            self.intent_classes_reverse = {v: k for k, v in self.intent_classes.items()}

        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def forward(
        self, 
        input: Text=None, 
        lowercase: bool=False, 
        rm_emoji: bool=True, 
        rm_url: bool=True, 
        rm_special_token: bool=False
    ):
        output = self.learner.process(
            sample=input,
            lowercase=lowercase,
            rm_emoji=rm_emoji,
            rm_url=rm_url,
            rm_special_token=rm_special_token
        )
        logits = output['intent'].get('intent_logits')

        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Performs temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        
        return logits / temperature

    def set_temperature(
        self, 
        valid_data: Union[Text, DataFrame], 
        text_col: str='text',
        intent_col: str='intent',
        tag_col: str='tags',
        shuffle: bool=False,
        lowercase: bool=True, 
        rm_emoji: bool=True, 
        rm_url: bool=True, 
        rm_special_token: bool=False
    ):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.

        :param valid_data: A Valid DataFrame or Valid path to .csv file.
        :param text_col: The column name of text data.
        :param intent_col: The column name of intent label data.
        :param tags_col: The column name of tags label data.
        :param lowercase: If True, lowercase data.
        :param rm_emoji: If True, replace the emoji token into <space> (" ").
        :param rm_url: If True, replace the url token into <space> (" ").
        :param rm_special_token: If True, replace the special token into <space> (" ").
                                 special token included of punctuation token, characters without vietnamese characters
        """
        # TODO: Processing data
        if isinstance(valid_data, Text):
            data_source = DenverDataSource.from_csv(test_path=valid_data,
                                                    text_col=text_col, 
                                                    intent_col=intent_col,
                                                    tag_col=tag_col, 
                                                    rm_emoji=rm_emoji, 
                                                    rm_url=rm_url, 
                                                    rm_special_token=rm_special_token, 
                                                    lowercase=lowercase)
            data_df = data_source.test.data
        else:
            data_df = valid_data

        # TODO: Shuffle data_df if shuffle is True
        if shuffle:
            data_df = data_df.sample(frac=1).reset_index(drop=True)

        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = ECELoss()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
    
        for i in tqdm(range(len(data_df))):
            input = data_df[text_col][i]
            label = data_df[intent_col][i]

            output = self.learner.process(
                sample=input,
                lowercase=lowercase,
                rm_emoji=rm_emoji,
                rm_url=rm_url,
                rm_special_token=rm_special_token
            )
            logits = output['intent'].get('intent_logits')
            label = self.intent_classes_reverse[label]

            logits_list.append(logits)
            labels_list.append(label)

        logits = torch.Tensor(logits_list).to(self.device)
        labels = torch.Tensor(labels_list).long().to(self.device)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        # before_temperature_ece = ece_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion.loss(logits.numpy(), labels.numpy(), 15)
        
        print(f"Before temperature - NLL: {before_temperature_nll:.3f}, ECE: {before_temperature_ece:.3f}")

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            
            return loss

        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        # after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        after_temperature_ece = ece_criterion.loss(
            self.temperature_scale(logits).detach().numpy(), labels.numpy(), 15)
        
        print(f"Optimal temperature: {self.temperature.item():.3f}")
        print(f"After temperature - NLL: {after_temperature_nll:3f}, ECE: {after_temperature_ece:.3f}")

        logits_sc = self.temperature_scale(logits)

        return logits, logits_sc, labels, self.temperature

    def save_model(self, model_dir: Text='./models', model_name: Text='model_with_temperature.tar.gz'):
        # TODO : Check path exists?
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        model_path = os.path.join(model_dir, model_name)
        torch.save(self.state_dict(), model_path)

        logging.info(f"The path to the save model with temperature: {model_path}")

    def load_model(self, model_path: Text=None):
        # TODO: Check the model file exists
        if not os.path.isfile(model_path):
            raise ValueError(f"The model file `{model_path}` is not exists or broken !")

        self.load_state_dict(torch.load(model_path))
        
        
    def visualize(self, logits, logits_sc, labels, save_dir: Text='./visualize', view_mode: bool=False):
        if type(logits) == torch.Tensor:
            logits = logits.cpu().numpy()
        if type(logits_sc) == torch.Tensor:
            logits_sc = logits_sc.detach().cpu().numpy()
        if type(labels) == torch.Tensor:
            labels = labels.cpu().numpy()

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        confidence_histogram = ConfidenceHistogram()
        ori_conf_hist = confidence_histogram.plot(
            logits, labels, title="Confidence Histogram - Original Model")
        ori_conf_hist.savefig(
            os.path.join(save_dir, 'conf_histogram_original_model.png'), bbox_inches='tight')
        if view_mode:
            ori_conf_hist.show()


        temp_scale_conf_hist = confidence_histogram.plot(
            logits_sc, labels, title="Confidence Histogram - Model with Temp. Scale")
        temp_scale_conf_hist.savefig(
            os.path.join(save_dir, 'conf_histogram_temperature_scale.png'), bbox_inches='tight')
        if view_mode:
            temp_scale_conf_hist.show()

        reliability_diagram = ReliabilityDiagram()
        ori_rel_diagram = reliability_diagram.plot(
            logits, labels, title="Reliability Diagram - Original Model")
        ori_rel_diagram.savefig(os.path.join(save_dir, 'rel_diagram_original_model.png'), bbox_inches='tight')
        if view_mode:
            ori_rel_diagram.show()

        temp_scale_rel_diagram = reliability_diagram.plot(
            logits_sc, labels, title="Reliability Diagram - Model with Temp. Scale")
        temp_scale_rel_diagram.savefig(os.path.join(save_dir, 'rel_diagram_temperature_scale.png'), bbox_inches='tight')
        if view_mode:
            temp_scale_rel_diagram.show()

        logging.info(f"Path to the visulization: {save_dir}")

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
        output = self.learner.process(sample, lowercase, rm_emoji, rm_url, rm_special_token)

        logits = output['intent'].get('intent_logits')
        temperature = self.temperature.detach().cpu().numpy()[0]
        logits_sc = logits / temperature

        probs = softmax(logits_sc)
        conf = np.max(probs)
        
        output['intent']['confidence'] = conf

        return output

