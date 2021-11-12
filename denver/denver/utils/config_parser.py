# -*- coding: utf-8
# Copyright (c) 2021 by Phuc Phan

import os
import re
import ast
import yaml
import logging
import collections
import configparser

from pathlib import Path
from typing import Text, Union

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class ConfigParserMultiValues(collections.OrderedDict):

    """A class to get config from file .ini"""

    def __setitem__(self, key, value):
        if key in self and isinstance(value, list):
            self[key].extend(value)
        else:
            super().__setitem__(key, value)

    @staticmethod
    def getlist(value):
        value = re.sub("[\[\]]", '', value)
        values = value.split(',')
        values = [value.strip() for value in values]
        return values


def get_config_section(filename, section):
    """Get config by sestion

    :param filename: Path to file .ini
    :param section: The section to take.

    :returns: A dict key, value from this section.
    """
    config = configparser.ConfigParser(strict=False, 
                                       empty_lines_in_values=False, 
                                       dict_type=ConfigParserMultiValues, 
                                       converters={"list": ConfigParserMultiValues.getlist})
    config.read([filename])

    config_parser = configparser.ConfigParser()
    config_parser.optionxform = str
    config_parser.read(filename)

    dict_session = {}

    # for section in config_parser.sections():
    for key in dict(config_parser.items(section)):
        values = config.getlist(section, key)
        
        if len(values) > 1:
            dict_session[key] = [ast.literal_eval(value) for value in values]
        else:
            dict_session[key] = ast.literal_eval(values[0])
    
    return dict_session

def get_config_yaml(config_file: Union[str, Text, Path]):
    """This function will parse the configuration file that was provided as a 
    system argument into a dictionary.

    :param config_file: Path to the config file

    :return: A dictionary contraining the parsed config file
    """
    if not isinstance(config_file, str):
            raise TypeError(f"The config must be a file path not {type(config_file)}")
    elif not os.path.isfile(config_file):
        raise FileNotFoundError(f"  File {config_file} is not found!")
    elif not config_file[-5:] == ".yaml":
        raise TypeError(f"We only support .yaml format")
    else:
        logger.info(f"Load config-file from: {config_file}")
        with open(config_file, 'r') as file:
            cfg_parser = yaml.load(file, Loader=yaml.Loader)

    return cfg_parser
