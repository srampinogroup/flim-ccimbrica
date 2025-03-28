flim-ccimbrica
##############

Machine learning analysis on FLIM measurements of Coccomyxa cimbrica exposed to Cu(II).

Install
*******

We assume you have Python 3.13 installed. It might work with other
versions as long as all packages in `requirements.txt` can be
installed.


Usage
*****

flim.py
=======

This is the main module of the project. It contains all functions
needed to read, curate and preprocess the raw data:
.. code-block:: python

  import flim

The raw data can be imported with:
.. code-block:: python

  raw_df = flim.read_flim_df()

the preprocessed data with:
.. code-block:: python

  processed_df = flim.load_processed_flim()

and the curated data set augmented with statistical, fit and
interaction features with:
.. code-block:: python

  df = flim.load_and_add_all()

If the module is run, it will produce a sample of the raw data set
and store it into `sampledf.txt`.

lr_test.py
==========

Runs the tests over the four chosen regression models:
`LinearRegresssion
<https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html>`_,
`Ridge
<https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`_,
`RandomForestRegressor
<https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html>`_
and `GradientBoostingRegressor
<https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html>`_.
