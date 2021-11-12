# -*- coding: utf-8
# Copyright (c) 2021 by Phuc Phan

import os
import pandas as pd

from denver.constants import *
from denver.utils.utils import load_json

def hiperopt_hiplot_cli(
    hiperopt_stats_path, 
    output_dir=None, 
    **kwargs
):
    """
    Produces a parallel coordinate plot about hyperparameter optimization
    creating one HTML file and optionally a CSV file to be read by hiplot

    :param hiperopt_stats_path: path to the hiperopt results JSON file
    :param output_dir: path where to save the output plots
    """

    fname = 'hiperopt_hiplot.html'
    fname_path = generate_fname_template_path(
        output_dir,
        fname
    )
    
    hiperopt_stats = load_json(hiperopt_stats_path)
    hiperopt_df = hiperopt_results_to_dataframe(
        hiperopt_stats['hiperopt_results'], 
        hiperopt_stats['hiperopt_config']['parameters'], 
        hiperopt_stats['hiperopt_config']['metric']
    )

    hiperopt_hiplot(
        hiperopt_df, 
        fname=fname_path
    )

def hiperopt_hiplot(
        hiperopt_df,
        fname,
):
    import hiplot as hip
    experiment = hip.Experiment.from_dataframe(hiperopt_df)
    experiment.to_html(fname)

def hiperopt_results_to_dataframe(
    hyperopt_results, 
    hyperopt_parameters, 
    metric
):
    df = pd.DataFrame(
        [
            {metric: res['metric_score'], **res['parameters']} for res in hyperopt_results
        ]
    )
    df = df.astype(
        {hp_name: hp_params[TYPE]
         for hp_name, hp_params in hyperopt_parameters.items()}
    )
    return df

def generate_fname_template_path(output_dir, fname_template):
    """Ensure path to template file can be constructed given an output dir.

    Create output directory if yet does exist.
    :param output_dir: Directory that will contain the filename_template file
    :param filename_template: name of the file template to be appended to the
            filename template path
    :return: path to filename template inside the output dir or None if the
             output dir is None
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, fname_template)
    return None
