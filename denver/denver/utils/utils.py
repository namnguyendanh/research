# -*- coding: utf-8
# Copyright (c) 2021 by Phuc Phan

import os
import re
import json
import logging
import datetime
import pandas as pd
import urllib.request

from pathlib import Path
from typing import Any, Union
from denver.utils.progessbar import MyProgressBar

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def tokenize(text):
    """Function tokenize a text
    
    :param text: A text (str)

    :returns: a list token after tokenize.
    """
    # return [x.lower() for x in nltk.word_tokenize(text)]
    return [ x.lower() for x in text.split() ]

def check_url_exists(txt):
    """Function to check url exists. Return True if exists, otherwise
    
    :param txt: The user's input.
    
    :returns: True if exists url, otherwise.
    """

    url_regex = re.compile(r'\bhttps?://\S+\b')

    return re.match(url_regex, txt) is not None


def download_url(url:str, dest:str, name:str, overwrite:bool=False):
    """Download the model from `url` and save it to `dest` with the name is `name`

    :param url: The url to download
    :param dest: The directory folder to save
    :param name: The name file to save
    :param overwrite: If True, overwrite the old file.
    """
    if not os.path.exists(dest):
        logger.debug(f"Folder {dest} does not exist. Create a new folder in {dest}")
        os.makedirs(dest)

    if os.path.exists(dest + name) and not overwrite:
        logger.debug(f"File `{dest + name}` already exists!")
        return

    logger.info(f"Downloading file from: {url}")
    try:
        urllib.request.urlretrieve(url, dest + name, MyProgressBar(name))
        logger.debug(f"Path to the saved file: `{dest + name}`")
    except:
        logger.error(f"Cann't download the file `{name}` from: `{url}`")

def ifnone(a: Any, b: Any)->Any:
    """`a` if `a` is not None, otherwise `b`."""
    return b if a is None else a

def rename_file(file_path: Union[str, Path]):
    """Function to change the name file into other name file
    
    :param file_path: The path to the file

    """
    time = re.sub('\.[0-9]*','_',str(datetime.datetime.now()))\
             .replace(" ", "_").replace("-", "").replace(":","").strip('_')        

    if os.path.exists(file_path):
        logger.debug(f"File {file_path} already exists !")
        extend = '.' + file_path.split('.')[-1]
        nfile = "_" + str(time) + extend
        old_path = file_path.replace(extend, nfile)
        os.rename(file_path, old_path)
        logger.debug(f"Rename file `{file_path.split('/')[-1]}` into file `{old_path.split('/')[-1]}`")


def convert_to_ner(entities, text):

    tokens = text.split(" ")
    list_text_label = ['O']*len(tokens)

    for info in entities:
        label = info['entity']

        start = info['start']
        end = info['end']

        value = text[start:end]
        list_value = value.split(" ")

        index = len(text[:start].split(" ")) - 1
        list_text_label[index] = 'B-' + str(label)
        for j in range(1, len(list_value)):
            try:
                list_text_label[index + j] = 'I-' + str(label)
            except Exception as e:
                print(str(e))
                print(text)
                print(entities)
    return ' '.join(list_text_label)

def convert_to_BIO(entities, text):
        """Function convert entities follow rasa format to BIO.

        :param entities: Entities rasa format, example: [{'start': 8, 'end': 23, 'value': '30x35x(38-50)cm', 'entity': 'ask_confirm#size', 'confidence': 0.9999693632125854, 'extractor': 'FlairSequenceTagger'}].
        :param text: Raw text, example: Có size 30x35x(38-50)cm ko Shop.

        :returns: BIO format: BIO format, example: O O B-ask_confirm#size O O.
        """

        list_text_label = []
        tokens = text.split(" ")

        for i in range(len(tokens)):
            list_text_label.append('O')

        for info in entities:
            label = info['entity']

            start = info['start']
            end = info['end']

            value = text[start:end]
            list_value = value.split(" ")

            index = len(text[:start].split(" ")) - 1
            list_text_label[index] = 'B-' + str(label)
            for j in range(1, len(list_value)):
                try:
                    list_text_label[index + j] = 'I-' + str(label)
                except Exception as e:
                    logger.error(f"{str(e)}")
                    logger.error(f"{text}")
                    logger.error(f"{entities}")

        return ' '.join(list_text_label)

def convert_to_denver_format(examples):
    """Function to convert data with json format of Rasa to dataframe format of Denver
    
    :params examples: Data with json format of Rasa

    :returns: data_df: Returns data with dataframe format of Denver. 
    """
    examples = examples['rasa_nlu_data']['common_examples']

    final_data = []
    for example in examples:
        intent = example['intent']
        entities = example['entities']
        text = example['text']

        final_data.append({"sentence": text, "ic": intent,
                           "ner": convert_to_ner(entities, text)})

    data_df = pd.DataFrame(final_data)
    
    return data_df

def load_json(data_fp):
    with open(data_fp, 'r') as input_file:
        data = json.load(input_file)
    return data


