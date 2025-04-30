#!/usr/bin/env python3
"""
The ``lr_test`` module mainly provides tests for different regression
models. Different sets of features and targets can be tested by using
the ``LRTest`` dataclass. The main function computes the tests used
in the article.
"""
import time
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pandas
jsonpickle_pandas.register_handlers()

import flim


RESULTS_FILE = "out/lr_results.json"


@dataclass
class LRTest(dict):
  """
  Dataclass for automation of regression tests with fields:
  - name: str is a free title,
  - df: pd.DataFrame is the dataset or subset to keep,
  - x_lbl: list[str] is the list of features,
  - y_lbl: str is the target label,
  - n_splits: int = 8 is the number of splits for cross-validation.
  See ``generate_tests`` function for usage.
  """
  name: str
  df: pd.DataFrame
  x_lbl: list[str]
  y_lbl: str
  n_splits: int = 8


def generate_tests(df: pd.DataFrame) -> list[LRTest]:
  """
  Generate the list of tests to perform on the dataset. First fixed
  concentration, then fixed exposure, and finally the product of
  both (dosage). See ``LRTest`` dataclass.
  """
  tests = []
  # # Fixed concentration tests
  # tests += [
  #   LRTest(
  #     f"concentration = {con} µg/ml",
  #     df[df["concentration"] == con],
  #     flim.FEATURES,
  #     "exposure",
  #   )
  #   for con in df["concentration"].unique()
  # ]

  # # Fixed exposure tests
  # tests += [
  #   LRTest(
  #     f"exposure = {exp} h",
  #     df[df["exposure"] == exp],
  #     flim.FEATURES,
  #     "concentration",
  #   )
  #   for exp in df["exposure"].unique()
  # ]

  # Dosage test
  tests += [
    LRTest(
      "dosage",
      df,
      flim.FEATURES,
      "dosage",
      5,
      # 10,
    ),
  ]

  return tests


def lr_test(df: pd.DataFrame) -> dict:
  """
  Test different models on training dataset. The models are stored in
  the ``MODELS`` dictionary in the ``flim`` module. The features are
  stored in the ``FEATURES`` list in the same module. They can be
  changed test-wise by changing the appropriate fields.

  Each model is cross-validated using ``RepeatedKFold`` for each
  test.
  """
  t0 = time.process_time()
  flim.log("Starting tests...")

  lr_results = {}

  for test in generate_tests(df):
    flim.log(test.name)

    x_train, x_test, y_train, y_test = train_test_split(
        test.df[flim.FEATURES], test.df[test.y_lbl],
        test_size=flim.TEST_SIZE,
        random_state=flim.RANDOM_STATE)

    lr_results[test.name] = {
        # "test": test,
        "y_lbl": test.y_lbl,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": {},
      }

    scores = pd.DataFrame(columns=["model",
                                   "R² mean", "R² std",
                                   "RMSE mean", "RMSE std",
                                   "time"])
    for name, model in flim.MODELS.items():
      t1 = time.process_time()
      kfold = RepeatedKFold(
          n_splits=test.n_splits,
          n_repeats=flim.N_REPEATS,
          random_state=flim.RANDOM_STATE)
      cv_r2 = cross_val_score(model, x_train, y_train, cv=kfold)
      cv_rmse = cross_val_score(model, x_train, y_train, cv=kfold,
                               scoring="neg_root_mean_squared_error")
      scores.loc[len(scores.index)] = [
          name,
          cv_r2.mean(), cv_r2.std(),
          cv_rmse.mean(), cv_rmse.std(),
          time.process_time() - t1]

      model.fit(x_train, y_train)
      y_pred = model.predict(x_test)

      lr_results[test.name]["y_pred"][name] = y_pred

    print(scores)

  flim.log(f"Tests done in {time.process_time() - t0} s.")

  return lr_results


def save_results(lr_res: dict) -> None:
  """
  Save the results of ``lr_test`` to a JSON file. File is
  overwritten. Path may be created.
  """
  flim.log(f"Serilizing to {RESULTS_FILE}...")
  json_ = jsonpickle.encode(lr_res, keys=True)
  Path(RESULTS_FILE).parent.mkdir(parents=True,
                                  exist_ok=True)
  with open(RESULTS_FILE, "w", encoding="UTF-8") as f:
    f.write(json_)

  flim.log("Serialization done.")


def load_results() -> dict:
  """
  Load the results of ``lr_test`` from a JSON file.
  """
  try:
    with open(RESULTS_FILE, "r", encoding="UTF-8") as f:
      json_ = f.read()
  except FileNotFoundError:
    flim.err("LR test results file not found, "
             "did you run lr_test.py?")
    raise

  return jsonpickle.decode(json_, keys=True)


def main() -> None:
  """
  Load the dataset and perform the regression tests. The results are
  stored as JSON.
  """
  df = flim.load_and_add_all()
  lr_res = lr_test(df)
  save_results(lr_res)


if __name__ == "__main__":
  main()
