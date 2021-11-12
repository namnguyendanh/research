# -*- coding: utf-8
# Copyright (c) 2021 by phucpx@ftech.ai

import re
import logging
import pandas as pd

from typing import Union
from pathlib import Path
from unicodedata import normalize as nl
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class BalanceLearn(object):
    """
    Balancing data for a DataFrame 
    """
    def __init__(self):
        """Initialize a BalanceLearn class. """
        super(BalanceLearn, self).__init__()

    @classmethod
    def subtext_sampling(
        cls, 
        data: Union[str, Path, pd.DataFrame], 
        size: int=None, 
        label_col: str='label',
        replace: bool=False,
    ):
        """Balancing a dataframe 

        :param data: A dataframe or a path to the .csv file.
        :param size: Number of items to sampling.
        :param label_col: The column of a dataframe to sampling data follow it. 
        :param replace: Allow or disallow sampling of the same row more than once. 

        """
        if type(data) == str or type(data) == Path:
            data_df = pd.read_csv(data, encoding='utf-8').dropna()
        else:
            data_df = data

        y = data_df[label_col]

        if size is None:
            size = y.value_counts().min()

        list_df = []
        for label in y.value_counts().index:
            samples = data_df[data_df[label_col] == label]
            
            if size > len(samples) and replace == False:
                samples = samples.sample(n=len(samples), replace=replace)
            else:
                samples = samples.sample(n=size, replace=replace)

            list_df.append(samples)
        
        data_df = pd.concat(list_df)

        return data_df

def normalize(
    text, 
    rm_emoji: bool=False, 
    rm_url: bool=False, 
    lowercase: bool=False, 
    rm_special_token: bool=False
):
    '''Function to normalize text
    
    :param text: The text to normalize
    :param lowercase: If True, lowercase data
    :param rm_emoji: If True, replace the emoji token into <space> (" ")
    :param rm_url: If True, replace the url token into <space> (" ")
    :param rm_special_token: If True, replace the special token into <space> (" ")

    :returns: txt: The text after normalize.        
    '''

    # Convert input to UNICODE utf-8
    try:
        txt = nl('NFKC', text)
        # lowercase
        if lowercase:
            txt = txt.lower().strip()
            
        # Remove emoji
        if rm_emoji:
            emoji_pattern = re.compile("["
                                    u"\U0001F600-\U0001F64F"  # emoticons
                                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                    "]+", flags=re.UNICODE)
            txt = emoji_pattern.sub(r" ", txt) 

        # Remove url, link
        if rm_url:
            url_regex = re.compile(r'\bhttps?://\S+\b')
            txt = url_regex.sub(r" ", txt)

        # Remove special token and duplicate <space> token
        if rm_special_token:
            txt = re.sub(r"[^a-z0-9A-Z*\sÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠẾếàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸýửữựỳỵỷỹ]", " ", txt)        
            txt = re.sub(r"\s{2,}", " ", txt)

    except Exception as e:
        logger.error(f"  {text}")
        raise ValueError(f"{e}")

    return txt.strip()

def standardize_df(
    df: pd.DataFrame, 
    text_col: Union[int, str]='text', 
    label_col: Union[int, str]=None,
    intent_col: Union[int, str]=None, 
    tag_col: Union[int, str]=None, 
):
    """Standardize a dataframe following the standardization format.

    :param df: A DataFrame
    :param text_col: The column name of text data
    :param label_col: The column name of label data
    :param intent_col: The column specify the label of intent with jointly task IC and NER
    :param tag_col: The column specify the label of tagging with jointly task IC NER NER
    
    :return: df: A standardized DataFrame
    """
    
    if intent_col and tag_col:
        df = pd.DataFrame({
            'text': df[text_col],
            'intent': df[intent_col],
            'tag': df[tag_col]
        })        
    elif label_col:
        df = pd.DataFrame({
            'label': df[label_col],
            'text': df[text_col]
        })
    else:
        df = pd.DataFrame({
            'text': df[text_col]
        })
    return df

def split_data(
    data: pd.DataFrame, 
    pct: float=0.1, 
    is_stratify: bool=False, 
    text_col: str='text',
    label_col: str='intent', 
    seed: int=123, 
    *kwargs
):
    """Function to split data into train and test set follow as the pct value
    
    :param data: A data DataFrame
    :param pct: The ratio to split train/test set
    :param is_stratify: If True, data is split in a stratified fashion, using this as the class labels.

    :returns: train_df: A train DataFrame dataset
    :returns: test_df: A test DataFrame dataset
    """
    data = data.dropna()
    if is_stratify:
        train_df, test_df = train_test_split(
            data, test_size=pct, stratify=data[label_col], random_state=seed)
    else:
        train_df, test_df = train_test_split(
            data, test_size=pct, random_state=seed)

    return train_df, test_df
