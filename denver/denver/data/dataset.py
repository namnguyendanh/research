# -*- coding: utf-8
# Copyright (c) 2021 by phucpx@ftech.ai

import logging
import numpy as np

from pandas import DataFrame
from denver.data.preprocess import normalize

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class DenverDataset(object):
    """A base Denver Dataset class. """
    def __init__(
        self,
        data: DataFrame=None,
        lowercase: bool=True, 
        rm_emoji: bool=True, 
        rm_url: bool=True, 
        rm_special_token: bool=False,
        shuffle: bool=True,
    ):
        """Initialize a base Denver Dataset class

        :param data: A DataFrame
        :param lowercase: If True, lowercase data
        :param rm_emoji: If True, replace the emoji token into <space> (" ")
        :param rm_url: If True, replace the url token into <space> (" ")
        :param rm_special_token: If True, replace the special token into <space> (" ")
        :param shuffle: If True, shuffle data
        """

        if data is not None:
            self.data = self.normalize_df(
                data, rm_emoji=rm_emoji, rm_url=rm_url, rm_special_token=rm_special_token, lowercase=lowercase)

        self.length = len(self.data)
        self.shuffle = shuffle
        if self.shuffle:
            logger.debug(f"Using shuffle.")
            self._shuffle_indicies()
    
    def __len__(self):
        """Get the length of data. """

        return len(self.data)

    def __getdf__(self):
        """ Get a DataFrame. """
        return self.data

    def get_sentences(self):
        """ Get list sentences in Dataset. """
        return self.data['text'].values.tolist()

    def get_labels(self):
        """Get list labels in Dataset."""
        if 'label' in self.data.columns:
            return self.data['label'].values.tolist()
        return [self.data['intent'].values.tolist(), self.data['tag'].values.tolist()]

    def normalize_df(self, data_df, rm_emoji, rm_url, rm_special_token, lowercase):
        '''Normalize text data frame
        
        :param data_df: A dataframe
        :param lowercase: If True, lowercase data
        :param rm_emoji: If True, replace the emoji token into <space> (" ")
        :param rm_url: If True, replace the url token into <space> (" ")
        :param rm_special_token: If True, replace the special token into <space> (" ")
        
        :returns: A dataframe after normalized.
        '''
        data_df = data_df.dropna()
        data_df["text"] = data_df["text"].apply(lambda x: normalize(x, rm_emoji=rm_emoji, 
                                    rm_url=rm_url, rm_special_token=rm_special_token, lowercase=lowercase))

        return data_df

    def _shuffle_indicies(self, seed: int=123):
        """Random permute a sequence and use `seed`. """
        np.random.seed(seed)
        self.indicies = np.random.permutation(self.length)
