# -*- coding: utf-8
# Copyright (c) 2021 by Phuc Phan

import torch.nn

class DenverModel(torch.nn.Module):
    """Abstract base class for all downstream task models in Denver, such as SequenceTagger and TextClassifier.
    Every new type of model must implement these methods."""
    def __init__(self):
        super(DenverModel, self).__init__()
