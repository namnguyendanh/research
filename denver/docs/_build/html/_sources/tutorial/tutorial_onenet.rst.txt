===============================
Tutorial: Building OneNet model
===============================

.. contents:: Table of Contents

- **Install library:**

.. parsed-literal::

    pip uninstall denver  # if existed.
    pip install http://minio.dev.ftech.ai/resources-denver-v0.0.2-75854855/denver-0.0.2b0-py3-none-any.whl
    

Training a model
---------------------

**1. Create a DenverDataSource**

.. admonition:: **NOTE**

    For the jointly models, for example OneNet model, to build a ``DenverDataSource`` from a ``.csv`` 
    file or a ``DataFrame``, we need to define ``intent_cols`` for the label of the IC task, and 
    ``tag_cols`` for the label of NER task. Instead of define ``label_cols`` for a separate model 
    like ``ULMFITClassifier`` or ``FlairSequenceTagger``.

- From ``csv`` file:

.. code-block:: python
    :linenos:

    from denver.data import DenverDataSource

    train_path = './data/cometv3/train.csv'
    test_path = './data/cometv3/test.csv'

    data_source = DenverDataSource.from_csv(train_path=train_path, 
                                            test_path=test_path, 
                                            text_cols='text',
                                            intent_cols='intent', 
                                            tag_cols='tag', 
                                            lowercase=True)


- From ``DataFrame`` file:

.. code-block:: python
    :linenos:
    
    from denver.data import DenverDataSource

    train_df = A DataFrame
    test_df = A DataFrame

    data_source = DenverDataSource.from_df(train_df=train_df, 
                                        test_df=test_df, 
                                        text_cols='text',
                                        intent_cols='intent', 
                                        tag_cols='tag', 
                                        lowercase=True)



**2. Train the model**

.. code-block:: python
    :linenos:

    from denver.learners import OnenetLearner
    from denver.trainers.trainer import ModelTrainer

    learn = OnenetLearner(mode='training', 
                        data_source=data_source, 
                        rnn_type='lstm', 
                        dropout=0.5,
                        bidirectional=True, 
                        hidden_size=200, 
                        word_embedding_dim=50, 
                        word_pretrained_embedding='vi-glove-50d', 
                        char_encoder_type='cnn', 
                        char_embedding_dim=30, 
                        num_filters=128, 
                        ngram_filter_sizes=[3], 
                        conv_layer_activation='relu')

    trainer = ModelTrainer(learn=learn)
    trainer.train(base_path='./models/onenet/', 
                model_file='denver-onenet.tar.gz', 
                learning_rate=0.001, 
                batch_size=64, 
                num_epochs=150)

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

    learn = OnenetLearner(mode='inference', model_path='./models/onenet/denver-onenet.tar.gz')

    data_path = './data/cometv3/test.csv'

    metrics = learn.evaluate(data=data_path, 
                            text_cols='text', 
                            intent_cols='intent', 
                            tag_cols='tag',
                            lowercase=True)


Get the prediction
--------------------

- Get prediction for a given sample

.. code-block:: python
    :linenos:

    from pprint import pprint

    ## inference a sample

    prediction = learn.predict(sample="xe day con mau vàng k sh", lowercase=True)
    pprint(prediction)

    output = learn.process(sample="xe day con mau vàng k sh", lowercase=True)
    pprint(output)

- Get the predictions from a ``DataFrame`` or a file ``.csv``

.. code-block:: python
    :linenos:
    
    ## Get predictions from a Dataframe or path to .csv

    data_path = './data/cometv3/test.csv'

    data_df = learn.predict_on_df(data=data_path, 
                                text_cols='text', 
                                intent_cols=None, 
                                tag_cols=None, 
                                lowercase=True)

    data_df.head()

In addition, you can also refer to the illustrative examples `here`_.

.. _`here`: https://gitlab.ftech.ai/nlp/research/denver_core/-/tree/develop/tutorials