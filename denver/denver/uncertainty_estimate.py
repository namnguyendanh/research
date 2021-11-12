# -*- coding: utf-8
# Copyright (c) 2021 by Phuc Phan

import torch
import numpy as np

from fastai.text import to_np

class UncertaintyEstimator:

    """
    Estimate uncertainty score for a sample point
    """

    def __init__(self, verbose):
        """
        Inititalize a uncertainty estimator class.
        """
        self.verbose = verbose

    def entropy(self, probs, softmax=False):
        """The method to estimate uncertainty score

        :param probs: The probabilities predict
        :param softmax: If True using softmax, (defalt=False)

        :returns: entropy: The prediction of a T*N*C tensor with  T: the number of samples, N: the batch size and C: the number of classes
        """
        probs = to_np(probs)
        prob = probs.mean(axis=0)

        entrop = - (np.log(prob) * prob).sum(axis=1)
        return entrop
    
    def get_uncertainty_score(self, method, model, sample, n_times=10):
        """
        Get the uncertainty score of the sample.
        
        :param method: The method to calculate uncertainty score.
        :param model: The selected model to predict.
        :param sample: The sample to calculate uncertainty score. 
        :param n_times: The times predction.

        :returns: results: A dict format as following: {
                            'text': sample,
                            'intent': iclass,
                            'uncertainty_score': value,
                            'method': method.__name__
                            }
        """

        classes = model.data.classes
        
        pred = model.predict_with_mc_dropout(sample, n_times=n_times)

        probs = [prob[2].view((1,1) + prob[2].shape) for prob in pred]
        probs = torch.cat(probs)
        
        indx = probs.mean(dim=0).squeeze(0).argmax()
        iclass = classes[indx]
        
        e = method(probs)

        results = {
            'text': sample,
            'intent': iclass,
            'uncertainty_score': e.item(),
            'method': method.__name__
            }
        
        return results