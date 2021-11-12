Get started in 30 Seconds


Denver is a `simple`, `easy-to-use` toolbox and library that provides SOTA models for `Language Understanding` (LU) tasks including two main components: `Intent Classification` (IC) and `Named Entities Recognition` (NER). 

Denver built on PyTorch that allow users to train, test, evaluate, get predictions deep learning models without the need to write code.

A programmatic API is also available in order to use Denver from python code.


**[DENVER DOCUMENTATION](https://phanxuanphucnd.github.io/denver/_build/html/index.html)**   

**[DENVER TUTORIAL](https://github.com/phanxuanphucnd/denver/tree/main/tutorials)**


# I. Prepare environment

If you don't have Anaconda3 in your system, then you can use the link below to get it done.

- [Install Anaconda3 on Ubuntu 18.04](https://www.digitalocean.com/community/tutorials/how-to-install-anaconda-on-ubuntu-18-04-quickstart)

- Create a enviroment: `conda create --name dender python=3.6`
  
- Then activate your new environment: `source activate denver`

# II. Prepare data

-  You need to prepare your data as a file `.csv format`.

# III. Starting

## 1. **Quickly**

### 1.1 Install

- Clone repo:  `git clone git@gitlab.ftech.ai:nlp/research/denver_core.git`

- Install:  `pip install .`

### 1.2 Command Line Interface (CLI)

Command | Effect |
--- | --- |
denver train | - Training a model. |
dender evaluate | - Evaluate the performance of a model. |
denver test | - Performs the inference of a text input. |
denver get-predict | - Get the predictions of a dataset with .csv format, and export to a file with .csv format. |
denver experiment | - Performs the experiment of a model with a dataset with .csv format. | 

:one: **Training a model**

This main command is:
```js
denver train 

```
This command trains a Denver model, consist of a IC model and a NER model. If you want to train a IC model, you must pass into `path_to_config_file` a path to the IC config-file as `ulmfit_configs.yaml` that you defined. Similar to a NER model, you also provide the path to the NER config-file as `flair_configs.yaml` into `path_to_config_file`. We are provided a default configs with SOTA model for IC and NER in folder `configs`.

Arguments:
```js
Required arguments:

  --config CONFIG_FILE, -c CONFIG_FILE        Path to the config file.

Optional arguments:
  
  --help, -h                                  Show this help message and exit.
```

- Example:

  - `denver train --config ./configs/ulmfit_configs.yaml`
  - `denver train --config ./configs/flair_configs.yaml`
  - `denver train --config ./configs/onenet_configs.yaml`

:two: **Evaluate a model**

This main command is:
```js
denver evaluate

```
Similar to the command `denver train`, this command evalute a Denver model, consist of a IC model and a NER model. If not use `--file` argument, the default data are taken from `test_path` in the config file.

Arguments:
```js
Required arguments:
  
  --config CONFIG_FILE, -c CONFIG_FILE        Path to the config file.

Optional arguments:

  --file DATA_FILE, -f DATA_FILE              Path to the data file with .csv format.
  --help, -h                                  Show this help message and exit.

```

- Example:

  - `denver evaluate --config ./configs/ulmfit_configs.yaml` 
  - `denver evaluate --config ./configs/ulmfit_configs.yaml --file ./data/test.csv` 
  - `denver evaluate --config ./configs/flair_configs.yaml`
  - `denver evaluate --config ./configs/flair_configs.yaml --file ./data/test.csv`
  - `denver evaluate --config ./configs/onenet_configs.yaml`
  - `denver evaluate --config ./configs/onenet_configs.yaml --file ./data/test.csv`


:three: **Inference**

This main command is:
```js
denver test

```
This command to test a text sample. Note that, we must be config the `model_dir` and `name` coresspoding to the used model in config file that you pass into `--config`.

Arguments:
```js
Required arguments:

  --input TEXT_INPUT, -in TEXT_INPUT            The text input.
  --config CONFIG_FILE, -c CONFIG_FILE          Path to the config file.

Optional arguments:

  --help, -h                                    Show this help message and exit.

```

- Example:

  - `denver test --input "xin chao viet nam" --config ./configs/ulmfit_configs.yaml`
  - `denver test --input "xin chao viet nam" --config ./configs/flair_configs.yaml`
  - `denver test --input "xin chao viet nam" --config ./configs/onenet_configs.yaml`

:four: **Get predictions**

This main command is:
```js
denver get-predict

```
This command is provided to make predictions for a dataset, and storages to a file `.csv`. The optional `--file` may be a path to the file need to predict or `default` is a data that taken from `test_path` in the config file.

Arguments:
```js
Required arguments:

  --config CONFIG_FILE, -c CONFIG_FILE          Path to the config file.

Optional arguments:

  --file DATA_FILE, -f DATA_FILE                Path to the data file with .csv format.
  --help, -h                                    Show this help message and exit.

```

- Example:

  - `denver get-predict --config ./configs/ulmfit_configs.yaml` 
  - `denver get-predict --config ./configs/ulmfit_configs.yaml --file ./data/data.csv` 
  - `denver get-predict --config ./configs/flair_configs.yaml`
  - `denver get-predict --config ./configs/flair_configs.yaml --file ./data/data.csv`
  - `denver get-predict --config ./configs/onenet_configs.yaml`
  - `denver get-predict --config ./configs/onenet_configs.yaml --file ./data/data.csv`

:five: **Experiment**

This main command is:
```js
denver experiment

```

This command is provided to experiment the model, allow use pass a data file with `.csv` format into `--file`. 

Arguments:
```js
Required arguments:
  
  --file DATA_FILE, -f DATA_FILE                  Path to the data file with .csv format.
  --config CONFIG_FILE, -c CONFIG_FILE            Path to the config file.

Optional arguments:

  --pct TEST_SIZE, -p TEST_SIZE                   The ratio to split train/test (default=0.1).
  --help, -h                                      Show this help message and exit.
```

- Example:

  - `denver experiment --config ./configs/ulmfit_config.yaml --file ./data/data.csv --pct 0.1` 
  - `denver experiment --config ./configs/flair_config.yaml --file ./data/data.csv --pct 0.1` 
  - `denver experiment --config ./configs/onenet.yaml --file ./data/data.csv --pct 0.1`

:six: **Use hiperopt**

This main command is:
```js
denver hiperopt

```

This command is provided to find the optimal hyper-parameters of the model

Arguments:
```js
Required arguments:
  
  --config CONFIG_FILE, -c CONFIG_FILE            Path to the config file.

Optional arguments:

  --help, -h                                      Show this help message and exit.
```

- Example:

  - `denver hiperopt --config ./configs/ulmfit_config.yaml` 
  - `denver hiperopt --config ./configs/flair_config.yaml` 
  - `denver hiperopt --config ./configs/onenet_configs.yaml`


## 2. **For development**

Please, refer:

- [Denver Documention](https://nlp.pages.gitlab.ftech.ai/research/denver_core/)

- [Denver Tutorial](https://gitlab.ftech.ai/nlp/research/denver_core/-/tree/develop/tutorials)
