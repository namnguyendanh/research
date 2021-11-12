# -*- coding: utf-8
# Copyright (c) 2021 by Phuc Phan

class DenverLearner:
    """Abstract base class for all downstream task in Denver, such as Sequence Tagger and Text Classification.
    Every new type of model must implement these methods."""

    def train(self):
        """Training the model. """
        raise NotImplementedError()

    def evaluate(self):
        """Evaluate the model. """
        raise NotImplementedError()
    
    def process(self):
        """Predict for a sample and return the rasa format output. """
        raise NotImplementedError()

    def predict(self):
        """Predicts the class labels for the given sample."""
        raise NotImplementedError()

    def save_model(self, save_file):
        """Save the model in the `save_file` path. """
        raise NotImplementedError()