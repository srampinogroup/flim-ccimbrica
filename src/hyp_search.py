#!/usr/bin/env python3
"""
Perform hyper-parameter search methods for the models. These methods
are provided as an example of how we computed the search for the
hyper-parameters, but more refinement of the parameters have been
done to obtain the values used in the models in the ``flim`` module.

"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
import flim


def hyp_ridge(df: pd.DataFrame) -> None:
  """
  Search for best hyper-parameters of Ridge.

  """
  model = flim.MODELS["Rid"]

  param_grid = {
    "solver": ["svd", "cholesky", "lsqr", "sag"],
    "alpha": np.logspace(-4, 2, 10),
    "fit_intercept": [True, False],
  }

  search_best_hyp(df, model, param_grid)


def hyp_forest(df: pd.DataFrame) -> None:
  """
  Search for best hyper-parameters of RandomForestRegressor.

  """
  model = flim.MODELS["For"]

  param_grid = {
    "n_estimators": [300, 500, 700],
    "max_depth": [5, 15, 20],
    "min_samples_split": [15, 20, 25],
    "min_samples_leaf": [6, 8, 10],
    "max_features": np.linspace(0.2, 0.4, 4),
  }

  search_best_hyp(df, model, param_grid)


def hyp_gradient(df: pd.DataFrame) -> None:
  """
  Search for best hyper-parameters of GradientBoostingRegressor.

  """
  model = flim.MODELS["GBR"]

  param_grid = {
    "learning_rate": np.logspace(-4, 0, 5),
    "n_estimators": [100, 200, 300],
  }

  search_best_hyp(df, model, param_grid)


def search_best_hyp(df: pd.DataFrame, model,
                    param_grid: dict) -> None:
  """
  Cross-validate the model with a GridSearchCV and print the best
  hyper-parameters and the R² score.

  """
  # xlbl = [
  #   "counts_avg", "counts_std", "counts_skew", "counts_tix",
  #   "fit_rate", "fit_const",
  #   "counts_avg*fit_rate", "counts_skew*fit_const"
  # ]
  xlbl = flim.FEATURES
  ylbl = "dosage"
  x_train, x_test, y_train, y_test = train_test_split(
      df[xlbl], df[ylbl],
      test_size=flim.TEST_SIZE,
      random_state=flim.RANDOM_STATE)

  cv = GridSearchCV(model, param_grid, scoring="r2",
                    cv=5, verbose=3, n_jobs=-1)
  cv.fit(x_train, y_train)
  best_model = cv.best_estimator_
  y_pred = best_model.predict(x_test)
  r2 = r2_score(y_test, y_pred)
  print(best_model)
  print(f"R² score: {r2}")


def main() -> None:
  """
  Perform the hyper-parameter search for all models.
  Note: there is no hyper-parameters in LinearRegression.

  """
  df = flim.load_and_add_all()

  hyp_ridge(df)
  hyp_forest(df)
  hyp_gradient(df)


if __name__ == "__main__":
  main()
