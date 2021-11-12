============================
Tutorial: Building IC models
============================

.. contents:: Table of Contents

ULMFITClassifier
================

This page explains how to develop an Intent Classification (IC) model. So far, we have only 
provided a model ULMFITClassifier. The steps for development are as follows:

- **Install library:**

.. parsed-literal::

    pip uninstall denver  # if existed.
    pip install http://minio.dev.ftech.ai/resources-denver-v0.0.2-75854855/denver-0.0.2b0-py3-none-any.whl
    
Training a model
---------------------

**1. Create a DenverDataSource**

.. admonition:: **NOTE**

    For the separate model, like ``ULMFITClassifier`` or ``FlairSequenceTagger``, to build a 
    ``DenverDataSource`` from a ``.csv`` file or a ``DataFrame``, we need to define ``label_cols``
    for the label of that task. 

- From csv file:

.. code-block:: python
    :linenos:

    from denver.data.data_source import DenverDataSource

    train_path = './data/salebot/train.csv'
    test_path = './data/salebot/test.csv'

    data_source = DenverDataSource.from_csv(train_path=train_path,
                                            test_path=test_path,
                                            text_cols='text',
                                            label_cols='intent',
                                            lowercase=True, 
                                            rm_special_token=True, 
                                            rm_url=True, 
                                            rm_emoji=True)
                                            
- From DataFrame:

.. code-block:: python
    :linenos:

    data_source = DenverDataSource.from_df(train_df=train_df, 
                                        test_df=test_df, 
                                        text_cols='text',
                                        label_cols='intent',
                                        lowercase=True, 
                                        rm_special_token=True, 
                                        rm_url=True, 
                                        rm_emoji=True)

**2. Fine-tuning**

- You can also fine-tune Langue model from any other corpus before fine-tune with train dataset (Optional):

.. parsed-literal::

    **corpus**
        data_split_1.txt
        data_split_2.txt
        ...
        data_split_N.txt

.. code-block:: python
    :linenos:

    from denver.trainers.language_model_trainer import LanguageModelTrainer

    data_folder = './data/data_babe/'
    lm_fns_path = ['./models/ic/vi_wt_babe', './models/ic/vi_wt_vocab_babe']
    
    lm_trainer = LanguageModelTrainer(pretrain='wiki')
    lm_trainer.fine_tuning_from_folder(data_folder=data_folder, 
                                       lm_fns=lm_fns_path,
                                       learning_rate=1e-2,
                                       num_epochs=10,
                                       batch_size=128)


- From train data:

Before starting to train the classify model, we must be fine-tuning language model with the training 
dataset that used to train the classify model.

.. code-block:: python
    :linenos:
    
    from denver.trainers.language_model_trainer import LanguageModelTrainer

    lm_trainer = LanguageModelTrainer(pretrain='babe')
    lm_trainer.fine_tuning_from_df(data_df=data_source.train.data,
                                batch_size= 128,
                                num_epochs=10,
                                learning_rate=1e-3,
                                moms=[0.8, 0.7],
                                drop_mult=0.5)

We have provided pretrained language models including wiki and babe. You can use by replacing 
``pretrain='babe'`` or ``pretrain='wiki'`` or ``pretrain=None`` or pass the list path to the pretrained 
language model same as `lm_fns_path`_ in the example above.

.. _`lm_fns_path`: ../denver.trainers.html#denver.trainers.language_model_trainer.LanguageModelTrainer.fine_tuning_from_folder

**3. Train the classify model**

.. code-block:: python
    :linenos:

    from denver.learners import ULMFITClassificationLearner
    from denver.trainers.trainer import ModelTrainer

    learn = ULMFITClassificationLearner(mode='training', data_source=data_source)

    trainer = ModelTrainer(learn=learn)
    trainer.train(base_path='./models/intent/', 
                model_file='denver.pkl', 
                learning_rate=2e-2, 
                batch_size=128, 
                num_epochs=14)


Evaluate a model
---------------------

Evaluate a model with a test dataset.

- Use the model after trained with test dataset in data_source:

.. code-block:: python
    :linenos:

    # evaluate the test set in data source
    metrics = learn.evaluate()

    from pprint import pprint
    pprint(metrics)

- Maybe, you can also evalute with any test dataset from .csv file:

.. code-block:: python
    :linenos:

    test_path = './data/test.csv'

    metrics = learn.evaluate(data=test_path, 
                            text_cols='text', 
                            label_cols='intent', 
                            lowercase=True, 
                            rm_special_token=True, 
                            rm_url=True, 
                            rm_emoji=True)

- Load model from a path:

.. code-block:: python
    :linenos:

    # Path to test dataset
    test_path = './data/test.csv'
    model_path = './models/denver-vicls.pkl'

    learn = ULMFITClassificationLearner(mode="inference", model_path=model_path)
    metrics = learn.evaluate(data=test_path, 
                            text_cols='text', 
                            label_cols='intent', 
                            lowercase=True, 
                            rm_special_token=True, 
                            rm_url=True, 
                            rm_emoji=True)


Get the prediction
--------------------

- Get prediction for a given sample

.. code-block:: python
    :linenos:

    text = "Làm bằng chất liệu j vậy shop"

    # Output
    prediction = learn.predict(sample=text, 
                            with_dropout=False, 
                            lowercase=True, 
                            rm_special_token=True, 
                            rm_url=True, 
                            rm_emoji=True)

    # Output to rasa-format 
    output = learn.process(sample=text, 
                        with_dropout=False, 
                        lowercase=True, 
                        rm_special_token=True, 
                        rm_url=True, 
                        rm_emoji=True)

- Get the predictions from a ``DataFrame`` or a file ``.csv``

.. code-block:: python
    :linenos:
    
    # Batch prediction
    data_df = learn.predict_on_df(data='./data/test.csv', 
                                text_cols='text', 
                                lowercase=True, 
                                rm_special_token=True, 
                                rm_url=True, 
                                rm_emoji=True)

    # Predicts each sample from a DataFrame
    data_df = model.predict_on_df_by_step(data=df, 
                                        text_cols='text', 
                                        lowercase=True, 
                                        rm_special_token=True, 
                                        rm_url=True, 
                                        rm_emoji=True)

    data_df.to_csv('out_file.csv', index=False, encoding='utf-8')

- In additional, we are provided an get ``uncertainty-score`` method, use as following:

.. code-block:: python
    :linenos:

    text = "Làm bằng chất liệu j vậy shop"
    
    uncertainty_score = learn.get_uncertainty_score(sample=text, n_times=10)

In addition, you can also refer to the illustrative examples `here`_.

.. _`here`: https://gitlab.ftech.ai/nlp/research/denver_core/-/tree/develop/tutorials