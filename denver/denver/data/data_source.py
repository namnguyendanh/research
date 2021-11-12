# -*- coding: utf-8
# Copyright (c) 2021 by phucpx@ftech.ai

import os
import logging
import tempfile
import pandas as pd

from pathlib import Path
from typing import Union
from pandas import DataFrame
from flair.data import Corpus
from flair.datasets import ColumnCorpus
from fastai.text import TextLMDataBunch, TextList

from denver.data.dataset import DenverDataset
from denver.data.preprocess import BalanceLearn
from denver.data.preprocess import standardize_df
from denver.utils.utils import download_url as dl

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class DenverDataSource(object):
    """A DenverDataSource class."""
    
    def __init__(
        self, 
        name: str='salebot', 
        train: DenverDataset=None, 
        test: DenverDataset=None,
        lowercase: bool=True
    ) -> None:
        """Initialize a DenverDataSource class

        :param train: A DenverDataset train
        :param test: A DenverDataset test
        """
        super(DenverDataSource, self).__init__()

        self.tempdir = tempfile.mkdtemp()

        self.train = train
        self.test = test
        self.lowercase = lowercase

        if not train and not test:
            if name == 'salebot':
                # Download a available data salebot from ours server
                urls = {
                    'train.csv': 'http://minio.dev.ftech.ai/fastaibase-v0.0.4-f17ecdb1/salebot_train.csv',
                    'test.csv': 'http://minio.dev.ftech.ai/fastaibase-v0.0.4-f17ecdb1/salebot_test.csv'
                }

                dest = './data/.salebot/'
                for name, url in urls.items():
                    dl(url, dest, name)

                self.train = self._add_dataset(data_path=dest + 'train.csv', rm_emoji=True, 
                                            rm_url=True, lowercase=True, 
                                            rm_special_token=True, shuffle=shuffle)

                self.test = self._add_dataset(data_path=dest + 'test.csv', rm_emoji=True, 
                                            rm_url=True, lowercase=True, 
                                            rm_special_token=True, shuffle=shuffle)
            else:
                raise ValueError(f"[ERROR] Dataset `{name}` is not in the corpus data that we supplied."
                                f"Please select in ['salebot', ] or pass a DataFrame or DenverDataset "
                                f"type into `train` or `test`.")

    def _add_dataset(
        self, 
        data_path: Union[str, Path], 
        lowercase: bool=True, 
        rm_emoji: bool=True, 
        rm_url: bool=True, 
        rm_special_token: bool=True,
        shuffle: bool = False
    ):
        data = pd.read_csv(data_path, encoding='utf-8', error_bad_lines=False)
        data = standardize_df(df=data, text_col='sentence', label_col='ic')

        denver_dataset = DenverDataset(data, rm_emoji=True, rm_url=True, lowercase=True, 
                                       rm_special_token=True, shuffle=shuffle)
        return denver_dataset
        

    @classmethod
    def from_csv(
        cls, 
        train_path: Union[str, Path]=None, 
        test_path: Union[str, Path]=None, 
        text_col: Union[int, str]=None,
        label_col: Union[int, str]=None,
        intent_col: Union[int, str]=None, 
        tag_col: Union[int, str]=None, 
        lowercase: bool=True, 
        rm_emoji: bool=False, 
        rm_url: bool=False, 
        rm_special_token: bool=False,
        balance: bool=False, 
        size: int=None,
        replace: bool=False,
        shuffle: bool = False
    ):
        '''Get DenverDataLoaders from file .csv format

        :param train_path: The path to train data
        :param test_path: The path to test data
        :param text_col: The column name of text data
        :param label_col: The column name of label data
        :param intent_col: The column specify the label of intent with jointly task IC and NER
        :param tag_col: The column specify the label of tagging with jointly task IC NER NER
        :param lowercase: If True, lowercase data
        :param rm_emoji: If True, replace the emoji token into <space> (" ")
        :param rm_url: If True, replace the url token into <space> (" ")
        :param rm_special_token: If True, replace the special token into <space> (" "), 
                                 special token included of punctuation token, characters without vietnamese characters
        :param balance: If True, balance train data
        :param size: Number of items to sampling
        :param replace: Allow or disallow sampling of the same row more than once
        :param shuffle: If True, shuffle data
        '''

        if (train_path is not None) and (train_path != ""):
            if not train_path.endswith(".csv"):
                raise ValueError(
                    f"File {train_path} is of invalid file format .csv!")
            train_path = os.path.abspath(train_path)     
        else:
            train_path = None

        if (test_path is not None) and (test_path != ""):
            if test_path and not test_path.endswith(".csv"):
                raise ValueError(f"File {test_path} is of invalid file format .csv!")
            test_path = os.path.abspath(test_path)
        else:
            test_path = None

        train_df = None
        test_df = None

        if train_path is None and test_path is None:
            raise ValueError(
                f"`train_path` or `test_path` must be different from `None` value.")

        if train_path is not None:
            logger.debug(f"Loaded the train data from: {train_path}")
            train_df = pd.read_csv(train_path, encoding='utf-8')

        if test_path is not None:
            logger.debug(f"Loaded the test data from: {test_path}")
            test_df = pd.read_csv(test_path, encoding='utf-8')

        return cls.from_df(train_df=train_df, test_df=test_df, 
                           text_col=text_col, label_col=label_col, 
                           intent_col=intent_col, tag_col=tag_col, 
                           rm_emoji=rm_emoji, rm_url=rm_url, lowercase=lowercase, 
                           rm_special_token=rm_special_token, shuffle=shuffle, 
                           balance=balance, size=size, replace=replace)

    @classmethod
    def from_df(
        cls, 
        train_df: DataFrame=None, 
        test_df: DataFrame=None, 
        text_col: Union[int, str]=None,
        label_col: Union[int, str]=None,
        intent_col: Union[int, str]=None, 
        tag_col: Union[int, str]=None, 
        lowercase: bool=True, 
        rm_emoji: bool=False, 
        rm_url: bool=False, 
        rm_special_token: bool=False, 
        balance: bool=False, 
        size: int=None,
        replace: bool=False,
        shuffle: bool=False,
    ):
        """Get DenverDataLoaders from DataFrame

        :param train_df: A DataFrame
        :param test_df: A DataFrame
        :param text_col: The column name of text data
        :param label_col: The column name of label data
        :param intent_col: The column specify the label of intent with jointly task IC and NER
        :param tag_col: The column specify the label of tagging with jointly task IC NER NER
        :param lowercase: If True, lowercase data
        :param rm_emoji: If True, replace the emoji token into <space> (" ")
        :param rm_url: If True, replace the url token into <space> (" ")
        :param rm_special_token: If True, replace the special token into <space> (" "), 
                                 special token included of punctuation token, characters without vietnamese characters
        :param balance: If True, balance train data
        :param size: Number of the sampled items
        :param replace: Allow or disallow sampling of the same row more than once
        :param shuffle: If True, shuffle data
        """
        if train_df is None and test_df is None:
            raise ValueError(
                f"`train_df` or `test_df` must be different from `None` value.")
                
        elif train_df is not None:
            if text_col and text_col not in train_df.columns:
                raise ValueError(
                    f"Column `{text_col}` not in data. Please check name colums data")

            if label_col and label_col not in train_df.columns:
                raise ValueError(
                    f"Column `{label_col}` not in data. Please check name colums data")

            if intent_col and intent_col not in train_df.columns:
                raise ValueError(
                    f"Column `{intent_col}` not in data. Please check name colums data")

            if tag_col and tag_col not in train_df.columns:
                raise ValueError(
                    f"Column `{tag_col}` not in data. Please check name colums data")

        elif test_df is not None:
            if text_col and text_col not in test_df.columns:
                raise ValueError(
                    f"Column `{text_col}` not in data. Please check name colums data")

            if label_col and label_col not in test_df.columns:
                raise ValueError(
                    f"Column `{label_col}` not in data. Please check name colums data")

            if intent_col and intent_col not in test_df.columns:
                raise ValueError(
                    f"Column `{intent_col}` not in data. Please check name colums data")

            if tag_col and tag_col not in test_df.columns:
                raise ValueError(
                    f"Column `{tag_col}` not in data. Please check name colums data")

        train = None
        test = None
        
        if train_df is not None:
            if balance:
                _label_col = label_col if label_col else intent_col
                train_df = BalanceLearn().subtext_sampling(
                                data=train_df, size=size, label_col=_label_col, replace=replace)

            train_df = standardize_df(df=train_df, text_col=text_col, label_col=label_col, 
                                      intent_col=intent_col, tag_col=tag_col)

            train = DenverDataset(train_df, rm_emoji=rm_emoji, rm_url=rm_url, lowercase=lowercase, 
                                rm_special_token=rm_special_token, shuffle=shuffle)

        if test_df is not None:
            test_df = standardize_df(df=test_df, text_col=text_col, label_col=label_col, 
                                     intent_col=intent_col, tag_col=tag_col)

            test = DenverDataset(test_df, rm_emoji=rm_emoji, rm_url=rm_url, lowercase=lowercase, 
                                rm_special_token=rm_special_token, shuffle=shuffle)

        return cls(train=train, test=test, lowercase=lowercase)

    def build_databunch(
        self, 
        batch_size: int=128, 
        mini_batch_chunk_size: int=10, 
        pct: int=0.2, 
        seed: int=42, 
        num_workers: int=1
    ):
        '''Build DataBunch for classifier

        :param batch_size (int): Batch size
        :param mini_batch_chunk_size: If mini-batches are larger than this number, 
                                      they get broken down into chunks of this size for 
                                      processing purposes
        :param pct: The percent for valid set
        :param seed: The number of seed

        :returns: A DataBunch type same as in FastAI
        '''
        if isinstance(self.train, DenverDataset):
            train_df = self.train.data
        else:
            train_df = self.train

        mini_batch_size = batch_size

        if len(train_df) <= batch_size:
            mini_batch_size = int(len(train_df) / mini_batch_chunk_size) + 2

        data_lm = TextLMDataBunch.from_df(train_df=train_df, valid_df=train_df, path="")
        
        databunch = (TextList.from_df(train_df, '', vocab=data_lm.vocab, cols='text')
                .split_by_rand_pct(pct, seed=seed)
                .label_from_df(cols='label')
                .databunch(bs=mini_batch_size, num_workers=num_workers))
            
        return data_lm.vocab, databunch

    def convert_to_bio_file(self, data, out_file='train.txt'):
        """Convert Dataframe to BIO-format and save to temporal file with name is `temp_output.txt`.

        """

        list_tokens = []
        list_labels = []

        for _, row in data.iterrows():
            sentence = row.text
            boi = row.label

            list_tokens.append(sentence.split())
            list_labels.append(boi.split())

        out_file = self.tempdir + '/' + out_file
        with open(out_file, 'w') as write_file:
            logger.debug(f"  Convert to file BIO-format... {out_file}")
            for i in range(len(list_labels)):
                for j in range(len(list_labels[i])):
                    try:
                        write_file.write(
                            list_tokens[i][j] + '\t' + list_labels[i][j] + '\n')
                    except Exception as e:
                        logger.warning(
                            f"Exception: {e} \n"
                            f"- Line: {i}: LIST_TOKENS ({len(list_tokens[i])}): {list_tokens[i]}; "
                            f"LIST_LABELS ({len(list_labels[i])}): {list_labels[i]}")

                write_file.write('\n')

        return out_file

    def build_corpus(self):
        """Get a Corpus dataset same as Corpus in Flair
        
        :returns: corpus: A Corpus dataset same as Corpus in Flair .
        """

        self.train_file = None
        if self.train is not None:
            self.train_file = self.convert_to_bio_file(
                data=self.train.data, out_file='train.txt')
        self.test_file = None
        if self.test is not None:
            self.test_file = self.convert_to_bio_file(
                data=self.test.data, out_file='test.txt')

        columns = {
            0: 'text',
            1: 'ner'
        }

        if self.train_file is not None:
            corpus: Corpus = ColumnCorpus(
                self.tempdir, columns, train_file=self.train_file, test_file=self.test_file)
        elif self.test_file is not None:
            corpus: Corpus = ColumnCorpus(
                self.tempdir, columns, train_file=self.test_file, test_file=self.test_file)
        else:
            raise ValueError(
                f"[ERROR] `self.data.train` or `self.data.test` is `None` values.")

        return corpus
