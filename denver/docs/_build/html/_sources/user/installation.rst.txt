============
Installation
============


.. contents:: Table of Contents


Prepare environment
===================

Follow `the installation instructions`_ for Anaconda here. Download and install Anaconda3 \
(at time of writing, `Anaconda3-2020.07`_). Then create a conda Python 3.6 enviroment for \
organizing packages used in Denver:

.. parsed-literal::

    conda create --name dender python=3.6.9

To use Python from the enviroment you just created, activate the enviroment with:

.. parsed-literal::

    conda activate denver

.. _`the installation instructions`: https://docs.continuum.io/anaconda/install/
.. _`Anaconda3-2020.07`: https://repo.anaconda.com/archive/


Install Denver
==============

- **Install from repository:**

.. parsed-literal::

    git clone git@github.com:phanxuanphucnd/denver.git

    cd denver

    pip install .

- **Install from library:**

.. parsed-literal::

    pip uninstall denver  # if existed.
    pip install dist/denver-3.0.2-py3-none-any.whl


Check Your Install
==================

To see if you'ave successfully installed Denver, try running ``check-install`` with:

.. parsed-literal::

    denver check-install

If ``Successfully!`` is returned, i.c you are successfully.