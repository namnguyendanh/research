==================
Configuration File
==================

.. contents:: Table of Contents

Basics
======

To quickly train, evaluate, and test models, we have provided the CLI for users to use. The 
need to effectively use them is that we have to configure the config file for the model, its
arguments and training parameters.

A config file has the following general structures:

- ``DATASET``: Define the data paths to use for processes.

    - ``train_path``: Define the path to training data.
    - ``test_path``: Define the path to test data.
    - ``text_cols``: Define the column text input in dataset.
    - ``label_cols``: Define the column label in dataset with the separate problem IC or NER.

    With OneNet model, ``label_cols`` is replaced into ``intent_cols`` and ``tag_cols``. For example, 
    like example `config onenet`_.

    - ``pre_processing``: Define the functions to pre-processing.

        - ``lowercase_token``: If True, lower token.
        - ``rm_special_token``: If True, remove special token, included of punctuation tokens, characters without vietnamese characters
        - ``rm_url``: If True, remove url format
        - ``rm_emoiji``: If True, remove emoiji token
        - ``balance``: If not False, balacing data. The params comprise: {'size': 300, 'replace': False}


- ``MODEL``: Defines the configuration used by the model, including the following components:

    - ``input_features``:

        - ``type``: Define the type of input.
        - ``level``: Define the level of input.
    
    - ``encoder``: 

        - ``name``: Defines the name of the model used
        - ``args``: Define the arguments of the used model. 
    
        Depending on the model, we need to define the appropriate arguments for it. For example, 
        for model ``ULMFITClassifier``, we need ``loss_func``, ``drop_mult``, ``average``, ``beta`` 
        arguments based on the model built. We have the definition for each model based on 
        `the following`_ for model ``ULMFITClassifier``.

    - ``output_features``:

        - ``type``: Define the type of output model. 
        
        For example, for IC, the type is ``class``, and for NER the type is ``ner``, and for the joint
        model, the type is a list of corrresponding types, for example with OneNet model, its type is
        [class, ner]

- ``TRAINING_MODEL``: Define the training parameters

    - ``base_path``: Define the dictionary to storages model and logging.
    - ``model_file``: Define the file name of model to save.
    - ``is_save_best_model``: If True, save the best model, otherwise, save the final model

    - ``hyper_params``:

        - ``learning_rate``: Define the learning rate value.
        - ``batch_size``: Define the batch size value.
        - ``num_epochs``: Define the number of epochs to train.

    Additionally, there are some special parameters for each model, such as ``momentums`` of 
    ULMFITClassifier model.

    You can find the specific parameters based on the ``train()`` function of each model. 
    For example, like `here`_ with ULMFITClassifier, same for other models.

.. _`the following`: ../denver/denver.learners.html#denver.learners.ulmfit_cls_learner.ULMFITClassificationLearner
.. _`here`: ../denver/denver.learners.html#denver.learners.ulmfit_cls_learner.ULMFITClassificationLearner.train
.. _`config onenet`: ../user/configs.html#onenet

Pretrained Lookup-Table
=======================

Pretrained Language Model
-------------------------

**Cheat Sheet**

==============  ===============================================================
Name            Description
==============  ===============================================================
``wiki``        - Pretrained LM with Vietnamese Wiki Corpus.
``babe``        - Pretrained LM with Vietnamese Wiki Corpus + Babe Corpus.
==============  ===============================================================

Pretrained Flair Embeddings
----------------------------

**Cheat Sheet**

==================================== ===========================================================
Name                                    Description
==================================== ===========================================================
``vi-forward-1024-wiki``             - | Forward-pretrained Embedding with Vietnamese Wiki 
                                       | Corpus
``vi-backward-1024-wiki``            - | Backward-pretrained Embedding with Vietnamese Wiki 
                                       | Corpus
``vi-forward-1024-babe``             - | Forward-pretrained Embedding with Vietnamese Wiki  
                                       | Corpus + Babe Corpus
``vi-backward-1024-babe``            - | Backward-pretrained Embedding with Vietnamese Wiki
                                       | Corpus + Babe Corpus
``vi-forward-1024-lowercase-wiki``   - | Forward-pretrained Embedding with Vietnamese lowercase
                                       | Wiki Corpus
``vi-backward-1024-lowercase-wiki``  - | Backward-pretrained Embedding with Vietnamese lowercase
                                       | Wiki Corpus
``vi-forward-1024-lowercase-babe``   - | Forward-pretrained Embedding with Vietnamese lowercase
                                       |  Wiki Corpus + Babe Corpus
``vi-backward-1024-lowercase-babe``  - | Backward-pretrained Embedding with Vietnamese lowercase
                                       | Wiki Corpus + Babe Corpus
``vi-forward-2048-lowercase-wiki``   - | Forward-pretrained Embedding with Vietnamese lowercase
                                       | Wiki Corpus
``vi-backward-2048-lowercase-wiki``  - | Backward-pretrained Embedding with Vietnamese lowercase
                                       | Wiki Corpus
``multi-forward``                    - Forward-pretrained Embedding with Multi-language
``multi-backward``                   - Backward-pretrained Embedding with Multi-language
``news-forward``                     - Forward-pretrained Embedding with English Corpus
``news-backward``                    - Backward-pretrained Embedding with English Corpus
==================================== ===========================================================

Pretrained Word Embeddings (Glove)
----------------------------------

**Cheat Sheet**

=================  ===============================================================
Name               Description
=================  ===============================================================
``vi-glove-50d``   - Pretrained LM with Babe Corpus.
``vi-glove-100d``  - Pretrained LM with Babe Corpus.
=================  ===============================================================

Examples
========

ULMFITClassifier
----------------

.. code-block:: python

    DATASET:
        # The path to train dataset
        train_path: ./data/cometv3/train.csv
        # The path to test dataset
        test_path: ./data/cometv3/test.csv
        # Define the column input name in DataFrame dataset 
        text_cols: text
        # Define the column label name in DataFrame dataset 
        label_cols: intent
        # Define function pre-processing
        pre_processing:
            # lower token
            lowercase_token: True
            # remove special token, included of punctuation token, 
            # characters without vietnamese characters
            rm_special_token: True
            # remove url
            rm_url: True
            # remove emoji token
            rm_emoji: True
            # if not Fasle, using balance data, 
            # the params comprise {'size': 300, 'replace': False}.
            balance: False #{'size': 300, 'replace': True}


    MODEL:
        input_features:
            type: text
            level: word
            # The pretrained language model in 'babe', 'wiki' with Vietnamese. 
            # You can setup to a `None` value. 
            pretrain_language_model: 'babe'
        encoder: 
            name: ULMFITClassifier
            args:
                # The dropout multiple
                drop_mult: 0.3
                # The average in 'binary', 'micro', 'macro', 'weighted' or None
                average: weighted
                # beta: Parameter for F-beta score
                beta: 1

        output_features:
            # Define tag type, examples as: class
            type: class
    

    TRAINING_MODEL:
        # The directory to save models
        base_path: ./models/intent
        # the file name of model 
        model_file: denver-viclass.pkl
        # Save model, if True, storages the best model, otherwise storages the final model
        is_save_best_model: True
        # The hyper-parameters to train model
        hyper_params:
            # The learning rate
            learning_rate: 2e-2
            # The batch size
            batch_size: 128
            # The number epochs to train
            num_epochs: 14

FlairSequenceTagger
-------------------

.. code-block:: python

    DATASET:
        # The path to train dataset
        train_path: ./data/cometv3/train.csv
        # The path to test dataset
        test_path: ./data/cometv3/test.csv
        # Define the column input name in DataFrame dataset 
        text_cols: text
        # Define the column label name in DataFrame dataset 
        label_cols: tag
        # Define function pre-processing
        pre_processing:
            # lower token
            lowercase_token: True
            # remove special token, included of punctuation token, 
            # characters without vietnamese characters
            rm_special_token: False
            # remove url
            rm_url: True
            # remove emoji token
            rm_emoji: True
            # if not Fasle, using balance data, 
            # the params included of {'size': 300, 'replace': False}.
            balance: False

    MODEL:
        input_features:
            # Define the column input name in DataFrame dataset 
            name: 'sentence'
            type: text
            level: word
            # Define the type embeddings to use
            embedding_types: bi-pooled_flair_embeddings
            # The pretrained embedding Path or str, 
            # maybe is a List[str] or List[Path] if embedding_types in *bi-xxx_embeddings*
            pretrain_embedding: ['vi-forward-1024-lowercase-babe', 'vi-backward-1024-lowercase-babe']

        encoder: 
            name: FlairSequenceTagger
            args:
                # use_rnn: if True use RNN layer, otherwise use word embeddings directly
                use_rnn: True
                # rnn_layers: The number of RNN layers
                rnn_layers: 1
                # hidden_size: number of hidden states in RNN
                hidden_size: 1024
                # dropout: dropout probability
                dropout: 0.0
                # word_dropout: word dropout probability
                word_dropout: 0.05
                # locked_dropout: locked dropout probability
                locked_dropout: 0.5
                # reproject_embeddings: if True, adds trainable linear map on top of embedding 
                # layer. If False, no map. if int, reproject embedding into (int) dims.
                reproject_embeddings: True
                # beta: Parameter for F-beta score
                beta: 1

        decoder:
            # use_crf: if True use CRF decoder, else project directly to tag space
            crf: True

        output_features:
            # Define the column label name in DataFrame dataset 
            name: ner
            # Define the type, examples as: ner
            type: ner


    TRAINING_MODEL:
        # The directory to save models
        base_path: ./models/ner
        # the file name of model 
        model_file: comet-viner.pt
        # Save model, if True, storages the best model, otherwise storages the final model
        is_save_best_model: True
        # The hyper-parameters to train model
        hyper_params:
            # learning_rate: learning rate,.
            learning_rate: 0.1
            # batch_size: Size of batches during training
            batch_size: 128
            # num_epochs: The number of epochs to train.
            num_epochs: 300

OneNet
--------

.. code-block:: python

    DATASET:
        # The path to train dataset
        train_path: ./data/cometv3/train.csv
        # The path to test dataset
        test_path: ./data/cometv3/test.csv
        # Define the column text input in DataFrame dataset 
        text_cols: text
        # Define the column intent label in DataFrame dataset 
        intent_cols: intent
        # Define the column tag label in DataFrame dataset 
        tag_cols: tag
        # Define function pre-processing
        pre_processing:
            # lower token
            lowercase_token: True
            # remove special token, included of punctuation token, 
            # characters without vietnamese characters
            rm_special_token: False
            # remove url
            rm_url: True
            # remove emoji token
            rm_emoji: True
            # if not Fasle, using balance data, 
            # the params included of {'size': 300, 'replace': False}.
            balance: False


    MODEL:
        input_features:
            type: text
            level: word

        encoder: 
            name: Onenet
            args:
                # the number of dropout
                dropout: 0.5
                # rnn type
                rnn_type: 'lstm'
                # if True, use bidirectional 
                bidirectional: True
                # the number of hidden size layer
                hidden_size: 200
                # the number of rnn layer
                num_layers: 2
                # the number of word embedding dimension
                word_embedding_dim: 50
                # The pretrained word embeding {'vi-glove-50d', 'vi-glove-100d'} 
                # or path to the word embedding
                word_pretrained_embedding: vi-glove-50d
                # the number of char embedding dimension
                char_embedding_dim: 30
                # the type of char encoder type
                char_encoder_type: cnn
                # the number of filters of cnn
                num_filters: 128
                # the ngram filter sizes
                ngram_filter_sizes: [3]
                # the activation of convolutional layer
                conv_layer_activation: relu

        output_features:
            # Define tag type, examples as: class
            type: [class, ner]
    

    TRAINING_MODEL:
        # The directory to save models
        base_path: ./models/onenet
        # the file name of model 
        model_file: denver-onenet.tar.gz
        # Save model, if True, storages the best model, otherwise storages the final model
        is_save_best_model: True
        # The hyper-parameters for training the classify model 
        hyper_params:
            # The learning rate
            learning_rate: 0.001
            # The batch size
            batch_size: 64
            # The number epochs to training
            num_epochs: 200