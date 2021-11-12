.. Denver documentation master file, created by
   sphinx-quickstart on Mon Aug 17 17:51:15 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Denver's Documentation!
==================================

Denver is a simple, easy-to-use toolbox and library that provides SOTA models for Language Understanding (LU) tasks 
including two main components: Intent Classification (IC) and Named Entities Recognition (NER).

Denver built on PyTorch that allow users to train, test, evaluate, get predictions deep learning models without the 
need to write code.

A programmatic API is also available in order to use Denver from python code.

.. toctree::
   :maxdepth: 2
   :caption: User Documentation

   user/introduction
   user/installation
   user/training_data
   user/cli
   user/configs

.. toctree::
   :maxdepth: 2
   :caption: Programatic API

   tutorial/tutorial_ic
   tutorial/tutorial_ner
   tutorial/tutorial_onenet

.. toctree::
   :maxdepth: 2
   :caption: Models Documentation

   models/ulmfit_cls
   models/flair_seq_tagger
   models/onenet
   models/experiment_result

.. toctree::
   :maxdepth: 4
   :caption: Utilities Documentation

   denver/denver

.. toctree::
   :maxdepth: 2
   :caption: Etc.

   etc/author

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
