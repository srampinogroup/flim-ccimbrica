flim-ccimbrica
##############

Machine learning analysis on FLIM measurements of Coccomyxa cimbrica exposed to Cu(II).

Data set
********

The full data set used is in file ``src/flimdf.json``. A sample is
shown in ``src/out/sampledf.txt``.

Install (conda)
***************

The best way to run the code is to clone the repository::

  git clone https://github.com/srampinogroup/flim-ccimbrica
  cd flim-ccimbrica

We recommend using conda and the conda-forge channel to install
dependencies. If you do not have Anaconda, the `miniforge
<https://conda-forge.org/docs/user/introduction/>`_ installer is the
simplest way to get started. If you already have conda, make sure to
use the conda-forge channel.

Creating the virtual environment
================================

If need be, you can create and activate a python virtual environment
using::

  conda create -n flim python=3.13.1 -y
  conda activate flim

Then install all dependencies from ``requirements.txt``::

  conda install -y --file=requirements.txt

or with ``pip``::

  pip install -r requirements.txt

You should now be able to run the python files::

  cd src
  ./flim.py

Usage
*****

flim.py as imported module
==========================

This is the main module of the project. It contains all functions
needed to read, curate and preprocess the raw data::

  import flim

The raw data can be imported with::

  raw_df = flim.read_flim_df()

the preprocessed data with::

  processed_df = flim.load_processed_flim()

and the curated data set augmented with statistical, fit and
interaction features with::

  df = flim.load_and_add_all()

The module also declares the array ``MODELS`` of the four chosen
regression models and hyper-parameters we used:
`LinearRegresssion
<https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_,
`Ridge
<https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`_,
`RandomForestRegressor
<https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_
and `GradientBoostingRegressor
<https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html>`_.

Other modules
=============

Please read the documentation in the docstring of each python module
in ``src``. Here is a brief description of each module:

``flim.py``
  Main module, read and curate data and is imported by most of the
  other modules.

``plot_util.py``
  Utility for plotting data, including setting up the default
  matplolib configuration.

``lr_test.py``
  Linear regression module. Defines and performs computation of
  *R²* scores for models defined in ``flim.py`` and write
  results to ``out/lr_results.json``.

``nn_test.py``
  Neural network module. Creates the neural network architecture
  and...
  

``fit_decay.py``
  Exponential decay fit.

``plot_data.py``
  Generates plots on the fly or from results of ``lr_test.py``.

``hyp_search.py``
  Code demonstrating the protocol we used to perform cross-validated
  grid search of hyper-parameters.
