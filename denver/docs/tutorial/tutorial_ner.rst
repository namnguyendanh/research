=============================
Tutorial: Building NER models
=============================

.. contents:: Table of Contents

FlairSequenceTagger
===================

This page explains how to develop an Named Entities Recognition (NER) model. So far, we have only 
provided a model FlairSequenceTagger. The steps for development are as follows:

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

    from denver.data import DenverDataSource

    ## Path to train data and test data
    train_path = './data/train.csv'
    test_path = './data/test.csv'

    data_source = DenverDataSource.from_csv(train_path=train_path, 
                                            test_path=test_path, 
                                            text_cols='text', 
                                            label_cols='tag', 
                                            lowercase=True)

- From DataFrame:

.. code-block:: python
    :linenos:

    data_source =  DenverDataSource.from_df(train_df=train_df,
                                            test_df=test_df,
                                            text_cols='text', 
                                            label_cols='tag', 
                                            lowercase=True)

**2. Create embeddings**

- Get pre-trained embeddings:

.. code-block:: python
    :linenos:

    from denver.embeddings import Embeddings

    embeddings = Embeddings(embedding_types='pooled_flair_embeddings',  
                            pretrain='vi-forward-1024-uncase-babe')

    embedding = embeddings.embed()

- Note, You can also fine-tuning language model as embedding from Other Corpus. The structure of a Folder Data as followings:

.. parsed-literal::

    **corpus**/
        **train**/
            train_split_1.txt
            train_split_2.txt
            ...
            train_split_X.txt

        test.txt
        valid.txt

.. code-block:: python
    :linenos:

    embedding = embeddings.fine_tuning(corpus_dir='./data/corpus', 
                                       model_dir='./models', 
                                       batch_size=32, 
                                       max_epoch=10,
                                       learning_rate=20,
                                       checkpoint=False)

**3. Training a Ner model**

.. code-block:: python
    :linenos:

    from denver.learners import FlairSequenceTaggerLearner

    learn = FlairSequenceTaggerLearner(mode='training', 
                                    data_source=data_source, 
                                    tag_type='ner', 
                                    embeddings=embedding,
                                    hidden_size=1024,
                                    rnn_layers=1,
                                    dropout=0.0, 
                                    word_dropout=0.05, 
                                    locked_dropout=0.5, 
                                    reproject_embeddings=2048, 
                                    use_crf=True)

    trainer = ModelTrainer(learn=learn)
    trainer.train(model_dir=model_dir, 
                model_file='denver.pt', 
                learning_rate=0.1, 
                batch_size=32, 
                num_epochs=300)


Evaluate a model
---------------------

Evaluate a model with a test dataset.

- Use the model after train:

.. code-block:: python
    :linenos:

    # evaluate the test set in data source 
    metrics = learn.evaluate()

    from pprint import pprint
    pprint(metrics) 

- Maybe, you can also evalute with any test dataset from .csv file.

.. code-block:: python
    :linenos:

    # Path to test dataset
    test_path = './data/test.csv'

    metrics = learn.evaluate(data=test_path, 
                            text_cols='text', 
                            label_cols='tag', 
                            lowercase=True)

- Load model from a path:

.. code-block:: python
    :linenos:

    # Path to test dataset
    test_path = './data/test.csv'
    model_path = './models/denver-viner.pt'

    learn = FlairSequenceTagger(mode='inference', model_path=model_path)

    metrics = learn.evaluate(data=test_path, 
                            text_cols='text', 
                            label_cols='tag', 
                            lowercase=True)


Get the prediction
---------------------

- Get prediction for a given sample

.. code-block:: python
    :linenos:

    text = 'shop có ghế ăn ko'

    # Output
    prediction = learn.predict(sample=text, lowercase=True)
    print(prediction)

    # Output to rasa-format 
    output = learn.process(sample=text, lowercase=True)

    from pprint import pprint
    pprint(output)

- Get the predictions from a ``DataFrame`` or a file ``.csv``

.. code-block:: python
    :linenos:
    
    # Get prediction
    data_df = learn.predict_on_df(data='./data/test.csv', 
                                        text_cols='sentence', 
                                        is_normalize=True)

    data_df.to_csv('outfile.csv', index=False, encoding='utf-8')

In addition, you can also refer to the illustrative examples `here`_.

.. _`here`: https://gitlab.ftech.ai/nlp/research/denver_core/-/tree/develop/tutorials