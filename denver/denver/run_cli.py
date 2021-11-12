# -*- coding: utf-8
# Copyright (c) 2021 by Phuc Phan

import os
import json
import click
import logging
import datetime

from pathlib import Path
from pprint import pformat
from typing import Union, Text

from denver.constants import *
from denver.api import DenverAPI
from denver.utils.print_utils import *
from denver import logger, DENVER_VERSION
from denver.utils.config_parser import get_config_yaml
from denver.hiperopt.run import hiperopt as hiperopt_func
from denver.hiperopt.visualize import hiperopt_hiplot_cli
from denver.utils.print_utils import view_table, print_denver


@click.group()
def entry_point():
    print_denver(message='Hello !', denver_version=DENVER_VERSION)
    pass


@click.command()
@click.option('--config', '-c',
              required=True,
              default='./configs/configure_ic.yaml',
              help='The file configures the model.')
@click.option('--debug',
              default=False,
              is_flag=True,
              help='Setup mode debug.')
def train(config: Union[str, Text], debug: bool):
    if debug:
        logger.setLevel(logging.DEBUG)

    now = datetime.datetime.now()

    DenverAPI.train(config=config, monitor=True)
    print_style_time(message="The time trained: {}".format(datetime.datetime.now() - now))

@click.command()
@click.option('--file', '-f',
              required=False,
              help='The path to the test file.')
@click.option('--config', '-c',
              required=True,
              help='The file configures the model.')
@click.option('--debug',
              default=False,
              is_flag=True,
              help='Setup mode debug.')
def evaluate(file: Union[str, Path], config: Union[str, Text], debug):
    if debug:
        logger.setLevel(logging.DEBUG)

    try:
        results = DenverAPI.evaluate(config=config, data_path=file)

        print_style_free(message='Results: ')
        print("")
        view_table(results)

    except Exception as e:
        raise ValueError(f"ERROR: {e}")

@click.command()
@click.option('--input', '-in',
              required=True,
              help='The text input.')
@click.option('--config', '-c',
              required=True,
              help='The file configures the model.')
@click.option('--debug',
              default=False,
              is_flag=True,
              help='Setup mode debug.')              
def test(input: Union[str, Text], config: Union[str, Path], debug: bool):
    if debug:
        logger.setLevel(logging.DEBUG)
    
    now = datetime.datetime.now()

    try:
        output = DenverAPI.test(config=config, input=input)
        
        print_style_free(message="Prediction: ")
        print(pformat(output))
        print_style_time(message="Inference Time: {}".format(datetime.datetime.now() - now))
    except Exception as e:
        raise ValueError(f"{e}")


@click.command()
@click.option('--file', '-f',
              required=False,
              help='The path to the test file.')
@click.option('--export', '-ex',
              required=False,
              default='./data', 
              help='The path to the test file.')
@click.option('--config', '-c',
              required=True,
              help='The file configures the model.')
@click.option('--debug',
              default=False,
              is_flag=True,
              help='Setup mode debug.')              
def get_predict(
    file: Union[str, Path], 
    config: Union[str, Path], 
    export: Union[str, Path], 
    debug: bool
):
    if debug:
        logger.setLevel(logging.DEBUG)

    try:
        data_df = DenverAPI.get_predict(config=config, data_path=file)        
        if not os.path.exists(export):
            os.makedirs(export)

        save_path = Path(export)/'predictions.csv'
        
        data_df.to_csv(save_path, encoding='utf-8-sig', index=False)
        logger.info(f"Path to the saved predicttions-file: {save_path}")

    except Exception as e:
        raise ValueError(f"{e}")


@click.command()
@click.option('--file', '-f',
              required=False,
              help='The path to the test file.')
@click.option('--pct', '-p',
              required=True,
              default=0.1,
              help='The ratio to split train/test dataset.')
@click.option('--config', '-c',
              required=True,
              help='The file configures the model.')
@click.option('--debug',
              default=False,
              is_flag=True,
              help='Setup mode debug.')
def experiment(file: Union[str, Path], pct: float, config: Union[str, Path], debug: bool):
    if debug:
        logger.setLevel(logging.DEBUG)

    try:
        _ = DenverAPI.experiment(config=config, dataset=file, pct=pct, verbose=True)
    except Exception as e:
        raise ValueError(f"{e}")

@click.command()
@click.option('--config', '-c',
              required=True,
              help='The file configures the model.')
@click.option('--debug',
              default=False,
              is_flag=True,
              help='Setup mode debug.')
def hiperopt(config: Union[str, Path], debug: bool):


    if debug:
        logger.setLevel(logging.DEBUG)

    try:
        if isinstance(config, str) or isinstance(config, Path):
            cp_config = get_config_yaml(config_file=config)

        results = hiperopt_func(config=config)
    
        base_path = cp_config[TRAINING_PARAMS].get(BASE_PATH)
        hiperopt_stats_path = Path(base_path)/'hiperopt_stats/hiperopt_statistics.json'
        output_dir = Path(base_path)/'visualize'
        hiperopt_hiplot_cli(
            hiperopt_stats_path=hiperopt_stats_path, 
            output_dir=output_dir)

        logger.info(
            f"Path to the visualize output file: {output_dir/'/hiperopt_hiplot.html'}")

        ## get optimal parameters
        MAX_SCORE = 0.0
        OPTIMAL_PARAMETERS = {}
        for i in range(len(results)):
            metric_score = float(results[i]['metric_score'][0])
            if metric_score > MAX_SCORE:
                OPTIMAL_PARAMETERS = results[i]['parameters']
                MAX_SCORE = metric_score

        print_style_notice(
            message=f"Metric score: {MAX_SCORE}")
        print_style_notice(
            message=f"Optimal parameters: \n{json.dumps(OPTIMAL_PARAMETERS, indent=4)} ")
    except Exception as e:
        raise ValueError(f"{e}")


# @click.command()
# @click.option('--from', '-f', 'from_',
#               required=True,
#               help='The input file to convert.')
# @click.option('--to', '-t', 'to_',
#               required=True,
#               help='The output file is converted, (.csv).')
# def convert(from_: Union[str, Path], to_: Union[str, Path]):
#     try:
#         if str(from_).endswith('.md'):
#             path = os.path.abspath(from_)
#             td = training_data.load_data(path)
#             output = td.as_json()
#             json_data = json.loads(output)
#         elif str(from_).endswith('.json'):
#             path = os.path.abspath(from_)
#             with open(path) as f:
#                 json_data = json.loads(f)
#         else:
#             raise TypeError(f"We only support convert with .json or .md format of Rasa.")
        
#         data_df = convert_to_denver_format(examples=json_data)
        
#         save_dir = os.path.abspath('/'.join(to_.split('/')[:-1]))
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)

#         data_df.to_csv(to_, encoding='utf-8', index=False)

#     except Exception as e:
#         raise ValueError(f"{e}")

@click.command()
def check_install():
    try:
        logger.info(f"Successfully!")
    except Exception as e:
        print(e)


# Command: train
entry_point.add_command(train)

# Command: evaluation
entry_point.add_command(evaluate)

# Command: Test
entry_point.add_command(test)

# Command: get-predict
entry_point.add_command(get_predict)

# Command: experiment
entry_point.add_command(experiment)

# Command: hiperopt
entry_point.add_command(hiperopt)

## Command: convert 
# entry_point.add_command(convert)

## Comaand: check
entry_point.add_command(check_install)


if __name__ == "__main__":
    entry_point()
