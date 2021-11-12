# -*- coding: utf-8
# Copyright (c) 2021 by Phuc Phan

import logging
import itertools

from typing import Dict, List, Any
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from seqeval.metrics import f1_score as seqeval_f1
from allennlp.training.metrics.metric import Metric
from seqeval.metrics import classification_report as seqeval_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from seqeval.metrics import precision_score as seqeval_precision, recall_score as seqeval_recall

from denver.utils.print_utils import plot_confusion_matrix

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class Metrics(object):
    def __init__(self, name, beta=1):
        self.name = name
        self.beta = beta

        self._tps = defaultdict(int)
        self._fps = defaultdict(int)
        self._tns = defaultdict(int)
        self._fns = defaultdict(int)


    def add_tp(self, class_name):
        self._tps[class_name] += 1

    def add_tn(self, class_name):
        self._tns[class_name] += 1

    def add_fp(self, class_name):
        self._fps[class_name] += 1

    def add_fn(self, class_name):
        self._fns[class_name] += 1

    def get_tp(self, class_name=None):
        if class_name is None:
            return sum([self._tps[class_name] for class_name in self.get_classes()])
        return self._tps[class_name]

    def get_tn(self, class_name=None):
        if class_name is None:
            return sum([self._tns[class_name] for class_name in self.get_classes()])
        return self._tns[class_name]

    def get_fp(self, class_name=None):
        if class_name is None:
            return sum([self._fps[class_name] for class_name in self.get_classes()])
        return self._fps[class_name]

    def get_fn(self, class_name=None):
        if class_name is None:
            return sum([self._fns[class_name] for class_name in self.get_classes()])
        return self._fns[class_name]

    def precision(self, class_name=None):
        if self.get_tp(class_name) + self.get_fp(class_name) > 0:
            return (
                self.get_tp(class_name)
                / (self.get_tp(class_name) + self.get_fp(class_name))
            )
        return 0.0

    def recall(self, class_name=None):
        if self.get_tp(class_name) + self.get_fn(class_name) > 0:
            return (
                self.get_tp(class_name)
                / (self.get_tp(class_name) + self.get_fn(class_name))
            )
        return 0.0

    def f_score(self, class_name=None):
        if self.precision(class_name) + self.recall(class_name) > 0:
            return (
                (1 + self.beta*self.beta)
                * (self.precision(class_name) * self.recall(class_name))
                / (self.precision(class_name) * self.beta*self.beta + self.recall(class_name))
            )
        return 0.0

    def accuracy(self, class_name=None):
        if (
            self.get_tp(class_name) + self.get_fp(class_name) + self.get_fn(class_name) + self.get_tn(class_name)
            > 0
        ):
            return (
                (self.get_tp(class_name) + self.get_tn(class_name))
                / (
                    self.get_tp(class_name)
                    + self.get_fp(class_name)
                    + self.get_fn(class_name)
                    + self.get_tn(class_name)
                )
            )
        return 0.0

    def micro_avg_f_score(self):
        return self.f_score(None)

    def macro_avg_f_score(self):
        class_f_scores = [self.f_score(class_name) for class_name in self.get_classes()]
        if len(class_f_scores) == 0:
            return 0.0
        macro_f_score = sum(class_f_scores) / len(class_f_scores)
        return macro_f_score

    def micro_avg_accuracy(self):
        return self.accuracy(None)

    def macro_avg_accuracy(self):
        class_accuracy = [
            self.accuracy(class_name) for class_name in self.get_classes()
        ]

        if len(class_accuracy) > 0:
            return sum(class_accuracy) / len(class_accuracy)

        return 0.0

    def micro_avg_precision(self):
        return self.precision(None)
    
    def macro_avg_precision(self):
        class_precision = [
            self.precision(class_name) for class_name in self.get_classes()
        ]

        if len(class_precision) > 0:
            return sum(class_precision) / len(class_precision)

        return 0.0
    
    def micro_avg_recall(self):
        return self.recall(None)
    
    def macro_avg_recall(self):
        class_recall = [
            self.recall(class_name) for class_name in self.get_classes()
        ]

        if len(class_recall) > 0:
            return sum(class_recall) / len(class_recall)

        return 0.0

    def get_classes(self) -> List:
        all_classes = set(
            itertools.chain(
                *[
                    list(keys)
                    for keys in [
                        self._tps.keys(),
                        self._fps.keys(),
                        self._tns.keys(),
                        self._fns.keys(),
                    ]
                ]
            )
        )
        all_classes = [
            class_name for class_name in all_classes if class_name is not None
        ]
        all_classes.sort()
        
        return all_classes


def get_metric(y_true, y_pred):
    
    """Function to get metrics evaluation.
    
    :param y_pred: Ground truth (correct) target values.
    :param y_true: Estimated targets as returned by a classifier.
    
    :returns: acc, f1, precision, recall
    """

    acc       = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted")
    recall    = recall_score(y_true, y_pred,  average="weighted")

    report = classification_report(y_true, y_pred)

    return acc, f1, precision, recall, report

class OnetNetMetrics(Metric):
    """A class Metics for OneNet model. """
    def __init__(self) -> None:
        """
        A class OneNetMetrics.
        """
        # These will hold per label span counts.
        self._true_positives = 0 
        self._false_positives = 0 
        self._true_negatives = 0
        self._false_negatives = 0 

        # self.tag_metrics = Metrics("tagger", beta=1)

        self.intent_predictions = []
        self.intent_labels = []

        self.tags_predictions = []
        self.tags_labels = []

        self.words = []


    def __call__(self,
                 nlu_predictions: List[Dict[str, Any]],
                 nlu_labels: List[Dict[str, Any]], 
                 intent_predictions: List[Any],
                 intent_labels: List[Any], 
                 tags_predictions: List[Any],
                 tags_labels: List[Any]):
        """
        :param predictions: ``torch.Tensor``, required. A tensor of predictions of shape (batch_size, sequence_length, num_classes).
        :param gold_labels: ``torch.Tensor``, required. A tensor of integer class label of shape (batch_size, sequence_length). 
                            It must be the same shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        """

        # TODO: NLU

        for prediction, label in zip(nlu_predictions, nlu_labels): 
            for dat in prediction:
                for tag, span in prediction[dat]:
                    if dat not in label or (tag, span) not in label[dat]:
                        self._false_positives += 1
                    else:
                        self._true_positives += 1

            for dat in label:
                for tag, span in label[dat]:
                    if dat not in prediction or (tag, span) not in prediction[dat]:
                        self._false_negatives += 1


        # TODO: INTENT
        self.intent_labels.extend(intent_labels)
        self.intent_predictions.extend(intent_predictions)

        # TODO: SEQUENCE TAGGERS
        self.tags_labels.extend(tags_labels)
        self.tags_predictions.extend(tags_predictions)

    def get_metric(self, reset: bool=False, mode: bool=False):
        """Function get metric, return a Dict per label containing following the span based metrics: \n
        precision : float \n
        recall : float \n
        f1-measure : float \n
        Additionally, an ``overall`` key is included, which provides the precision, \n
        recall and f1-measure for all spans. 
        """
        # Compute the precision, recall and f1 for all spans jointly.
        _, _, f1 = self._compute_metrics(self._true_positives,
                                                      self._false_positives,
                                                      self._false_negatives)

        metrics = {}
        if mode:
            intent_classes = list(set(self.intent_labels))
            logger.debug(f"The numbers of intent classes: {len(intent_classes)}")
            ic_acc, ic_f1, ic_precision, ic_recall, ic_report = get_metric(self.intent_labels, self.intent_predictions)

            try:
                cm = confusion_matrix(self.intent_labels, self.intent_predictions, labels=intent_classes)
                plot_confusion_matrix(cm, target_names=intent_classes, title="cm_onenet", save_dir='./evaluation')
            except Exception as e:
                logger.warning(f"WARNING: {e}")

            tag_precision = seqeval_precision(self.tags_labels, self.tags_predictions)
            tag_f1 = seqeval_f1(self.tags_labels, self.tags_predictions)
            tag_recall = seqeval_recall(self.tags_labels, self.tags_predictions)
            tag_report = seqeval_report(self.tags_labels, self.tags_predictions)

            # tag_f1_micro = self.tag_metrics.micro_avg_f_score()
            # tag_precision = self.tag_metrics.precision()
            # tag_recall = self.tag_metrics.recall()

            metrics["intent"] = {
                'accucary': round(ic_acc, 4), 
                'f1-score': round(ic_f1, 4), 
                'precision': round(ic_precision, 4),
                'recall': round(ic_recall, 4)
            }
            metrics["cls_detailed"] = ic_report

            metrics["tags"] = {
                'f1-score': round(tag_f1, 4), 
                'precision': round(tag_precision, 4),
                'recall': round(tag_recall, 4)
            }

            metrics["tags_detailed"] = tag_report

        metrics["main_score"] = f1
        
        if reset:
            self.reset()
        return metrics

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = 0 
        self._false_positives = 0 
        self._false_negatives = 0 

        self.intent_predictions = []
        self.intent_labels = []

        self.tags_labels = []
        self.tags_predictions = []

        # self.tag_metrics = Metrics(name="tagger")


