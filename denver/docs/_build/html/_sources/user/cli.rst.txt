======================
Command Line Interface
======================

To use the Command Line Interface (CLI), we need a config file. You can copy the link below and 
download it or see the instructions in the ``Configuration File`` tab.

- Download config files: 

  - `onenet config`_ : http://minio.dev.ftech.ai/resources-denver-extension-8765e803/onenet_config.yaml
  - `ulmfit config`_ : http://minio.dev.ftech.ai/resources-denver-extension-8765e803/ulmfit_config.yaml
  - `flair config`_ : http://minio.dev.ftech.ai/resources-denver-extension-8765e803/flair_config.yaml

.. _`onenet config`: http://minio.dev.ftech.ai/resources-denver-extension-8765e803/onenet_config.yaml
.. _`ulmfit config`: http://minio.dev.ftech.ai/resources-denver-extension-8765e803/ulmfit_config.yaml
.. _`flair config`: http://minio.dev.ftech.ai/resources-denver-extension-8765e803/flair_config.yaml

.. contents:: Table of Contents

**Cheat Sheet**

=======================  =====================================================================
Command                  Effect
=======================  =====================================================================
``denver train``         - Training a model.
``denver evaluate``      - Evaluate the performance a model.
``denver test``          - Performs the inference of a text input.
``denver get-predict``   - | Get the predictions of a dataset with .csv format, and export to
                           | a file with .csv format.
``denver experiment``    - Performs the experiment of a model with a dataset with .csv format.
``denver hiperopt``      - Hyper-parameters optimization
=======================  =====================================================================



Training a model
================

**This main command is:**

.. parsed-literal::

  denver train 

This command trains a Denver model, consist of a IC model and a NER model. If you want to train 
a IC model, you must pass ``CONFIG_FILE`` into a path to the file ``ic_configs.yaml``
that you defined. Similar to a NER model, you also provide the path ``ner_configs.yaml`` to 
``CONFIG_FILE``. We are provided a default configs with SOTA model for IC and NER in 
folder ``configs``. 

The config file defines the model and its arguments; the model training parameters and the data 
used to train, evaluate, test the model. You can also refer to details `here`_.

.. _`here`: ../configs/ic.html

.. admonition:: **Arguments**

  .. parsed-literal::
    Required-arguments:

      --config CONFIG_FILE, -c CONFIG_FILE        Path to the config file.

    Optional-arguments:
      
      --help, -h                                  Show this help message and exit.

  - Example:

    - ``denver train --config ./configs/ic_configs.yaml``
    - ``denver train --config ./configs/ner_configs.yaml``


Evaluate a model
================

**This main command is:**

.. parsed-literal::
  denver evaluate

Similar to the command ``denver train``, this command evalute a Denver model, consist of a IC 
model and a NER model. If not use ``--file`` argument, the default data are taken from ``test_path``
in the config file.

.. admonition:: **Arguments**

  .. parsed-literal::
    Required-arguments:
  
      --config CONFIG_FILE, -c CONFIG_FILE        Path to the config file.

    Optional-arguments:

      --file DATA_FILE, -f DATA_FILE              Path to the data file with .csv format.
      --help, -h                                  Show this help message and exit.

  - Example:

    - ``denver evaluate --config ./configs/ic_configs.yaml``
    - ``denver evaluate --config ./configs/ic_configs.yaml --file ./data/test.csv``
    - ``denver evaluate --config ./configs/ner_configs.yaml``
    - ``denver evaluate --config ./configs/ner_configs.yaml --file ./data/test.csv``

Inference
=========

**This main command is:**

.. parsed-literal::
  denver test

This command to test a text sample. Note that, we must be config the ``model_dir`` and ``name`` 
coresspoding to the used model in config file that you pass into ``--config``.

.. admonition:: **Arguments**

  .. parsed-literal::
    Required-arguments:

      --input TEXT_INPUT, -in TEXT_INPUT            The text input.
      --config CONFIG_FILE, -c CONFIG_FILE          Path to the config file.

    Optional-arguments:

      --help, -h                                    Show this help message and exit.

  - Example:

    - ``denver test --input "xin chao viet nam" --config ./configs/ic_configs.yaml``
    - ``denver test --input "xin chao viet nam" --config ./configs/ner_configs.yaml``


Get predictions
===============

**This main command is:**

.. parsed-literal::
  denver get-predict

This command is provided to make predictions for a dataset, and storages to a file ``.csv``.
The optional ``--file`` may be a path to the file need to predict or ``default`` is a data 
that taken from ``test_path`` in the config file.

.. admonition:: **Arguments**

  .. parsed-literal::
    Required-arguments:

      --config CONFIG_FILE, -c CONFIG_FILE          Path to the config file.

    Optional-arguments:

      --file DATA_FILE, -f DATA_FILE                Path to the data file with .csv format.
      --help, -h                                    Show this help message and exit. 

  - Example:

    - ``denver get-predict --config ./configs/ic_configs.yaml``
    - ``denver get-predict --config ./configs/ic_configs.yaml --file ./data/data.csv``
    - ``denver get-predict --config ./configs/ner_configs.yaml``
    - ``denver get-predict --config ./configs/ner_configs.yaml --file ./data/data.csv``

Experiment
==========

**This main command is:**

.. parsed-literal::
  denver experiment


This command is provided to experiment the model, allow use pass a data file with ``.csv`` format 
into ``--file``.

.. admonition:: **Arguments**

  .. parsed-literal::
    Required-arguments:
      
      --file DATA_FILE, -f DATA_FILE                  Path to the data file with .csv format.
      --config CONFIG_FILE, -c CONFIG_FILE            Path to the config file.

    Optional-arguments:

      --pct TEST_SIZE, -p TEST_SIZE                   The ratio to split train/test (=0.1).
      --help, -h                                      Show this help message and exit.

  - Example:

    - ``denver experiment --config ./configs/ic_configs.yaml --file ./data/data.csv --pct 0.1``
    - ``denver experiment --config ./configs/ner_configs.yaml --file ./data/data.csv --pct 0.1``

Hiperopt
========

.. parsed-literal::
  denver hiperopt

This command is provided to find the optimal hyper-parameters of the model.

.. admonition:: **Arguments**

  .. parsed-literal::
    Required-arguments:
      
      --config CONFIG_FILE, -c CONFIG_FILE            Path to the config file.

    Optional-arguments:

      --help, -h                                      Show this help message and exit.

  - Example:

    - `denver hiperopt --config ./configs/ulmfit_config.yaml` 
    - `denver hiperopt --config ./configs/flair_config.yaml` 
    - `denver hiperopt --config ./configs/onenet_configs.yaml`

