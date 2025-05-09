#!/usr/bin/env python3
"""
This module tests neural networks on the same tabular dataset as the
module ``lr_test``. It is the only module needing Keras. K-fold
cross-validation is performed to compute the R² score.
"""
import time
import os
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error as mae_score
from sklearn.preprocessing import StandardScaler
# Remove next two lines to use default tensorflow backend
import jax # for requirements.txt
os.environ["KERAS_BACKEND"] = "jax"
import keras
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.callbacks import EarlyStopping

import flim
# import plot_data


RESULTS_FILE = "out/nn_results.json"
FOLDS_FILE = "out/nn_folds.json"


keras.utils.set_random_seed(flim.RANDOM_STATE)


# @dataclass
# class NNTest(dict):
#   """
#   Dataclass for automation of neural network tests with fields:
#   - name: str is a free title,
#   - df: pd.DataFrame is the dataset or subset to keep,
#   - x_lbl: list[str] is the list of features,
#   - y_lbl: str is the target label,
#   - n_splits: int = 8 is the number of splits for cross-validation.
#   See ``generate_tests`` function for usage.
#   """
#   name: str
#   df: pd.DataFrame
#   x_lbl: list[str]
#   y_lbl: str
#   n_splits: int = 8


def nn_exploration(df: pd.DataFrame) -> None:
  """
  Build a neural network and test accuracy with K-fold
  cross-validation using same features as ``lr_test``.
  """
  t0 = time.process_time()
  # xlbl = flim.FEATURES
  xlbl = ["counts_max", "counts_std", "counts_skew", "counts_tix",
          "counts_avg", "fit_rate", "fit_const"]
  ylbl = "dosage"

  r2s = []

  # n_epochs = 1400
  n_epochs = 3000
  # batch_sizes = [20, 60, 80, 90]
  batch_size = 80
  # learning_rates = np.linspace(0.0034, 0.0036, 5)
  learning_rate = 0.0035
  # weight_decays = np.linspace(0.0013, 0.0025, 30)
  weight_decay = 0.0017
  # augment_props = np.linspace(0, 1, 5)
  prop = 0.0
  # n_neurons = [70, 80, 90]
  n_neuron = 80
  # dropouts = np.insert(np.linspace(0.075, 0.15, 24), 0, 0)
  # dropout = 0.1
  dropout = 0

  folds_df = pd.DataFrame(columns=["r2_score",
                                   "val_r2_score",
                                   "test_r2_score",
                                   "fold"])

  for _dummy in [1]:
  # for batch_size in batch_sizes:
  # for learning_rate in learning_rates:
  # for weight_decay in weight_decays:
  # for prop in augment_props:
  # for n_neuron in n_neurons:
  # for dropout in dropouts:
    df = flim.augment_dataset(df, prop)

    x = df[xlbl]
    y = df[ylbl]

    std = StandardScaler()
    x = std.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=flim.TEST_SIZE,
        random_state=flim.RANDOM_STATE)

    optimizer = keras.optimizers.AdamW(learning_rate,
                                       weight_decay=weight_decay)
    callbacks = [EarlyStopping(monitor="val_r2_score", patience=10)]
    # callbacks = None

    model = Sequential()
    model.add(Input(shape=(x_train.shape[1],)))
    model.add(Dense(n_neuron, activation="relu"))
    model.add(Dropout(dropout))
    # model.add(Dense(n_neuron, activation="relu"))
    # model.add(Dropout(dropout))
    model.add(Dense(1, activation="relu")) # target is always > 0

    model.compile(loss="mean_squared_error",
                  metrics=["r2_score"],
                  optimizer=optimizer)
    model.summary()

    n_splits = 5
    kfold = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=flim.RANDOM_STATE)
    loss = 0
    r2_tot = 0

    for i, (i_train, i_test) in enumerate(kfold.split(x, y)):
      flim.log(f"Split {i} / {n_splits}")
      x_train = x[i_train, :]
      y_train = y.iloc[i_train]
      x_val = x[i_test, :]
      y_val = y.iloc[i_test]
      hist = model.fit(x_train, y_train,
                       epochs=n_epochs,
                       batch_size=batch_size,
                       validation_data=(x_val, y_val),
                       callbacks=callbacks,
                       verbose=2)

      loss += model.evaluate(x_test, y_test, verbose=0)[0] \
              / n_splits
      y_pred = model.predict(x_test)
      r2 = r2_score(y_test, y_pred)
      r2_tot += r2 / n_splits

      fold_df = pd.DataFrame(np.column_stack(
        (hist.history["r2_score"], hist.history["val_r2_score"])),
        columns=["r2_score", "val_r2_score"])
      fold_df["test_r2_score"] = r2
      fold_df["fold"] = i

      folds_df = pd.concat([folds_df, fold_df], ignore_index=True)

    flim.log(f"{n_splits}-fold R², loss")
    print(r2_tot, loss)
    r2s += [r2_tot]

  y_pred = model.predict(x_test)
  pdf = pd.DataFrame({"y_test": y_test.values,
                      "y_pred": y_pred[:, 0].tolist()})
  pdf.sort_values(by="y_test", inplace=True)

  print(pdf.sample(10))
  flim.log("R²")
  print(r2_score(pdf["y_test"], pdf["y_pred"]))
  flim.log("MAE")
  print(mae_score(pdf["y_test"], pdf["y_pred"]))

  save_results(pdf, folds_df)

  flim.log(f"Done in {time.process_time() - t0:.3f} s.")


def save_results(pdf: pd.DataFrame, folds_df: pd.DataFrame) -> None:
  """
  Save the results of ``nn_test`` to JSON files. Files are
  overwritten. Path may be created.
  """
  flim.log(f"Serilizing to {RESULTS_FILE}...")
  Path(RESULTS_FILE).parent.mkdir(parents=True,
                                  exist_ok=True)
  with open(RESULTS_FILE, "w", encoding="UTF-8") as f:
    pdf.to_json(f)

  flim.log(f"Serilizing to {FOLDS_FILE}...")
  Path(FOLDS_FILE).parent.mkdir(parents=True,
                                  exist_ok=True)
  with open(FOLDS_FILE, "w", encoding="UTF-8") as f:
    folds_df.to_json(f)

  flim.log("Serialization done.")


def load_results() -> (pd.DataFrame, pd.DataFrame):
  """
  Load the results of ``nn_test`` from JSON files.
  """
  pdf: pd.DataFrame
  folds_df: pd.DataFrame

  try:
    with open(RESULTS_FILE, "r", encoding="UTF-8") as f:
      pdf = pd.read_json(f)
    with open(FOLDS_FILE, "r", encoding="UTF-8") as f:
      folds_df = pd.read_json(f)
  except FileNotFoundError:
    flim.err("LR test results file not found, "
             "did you run nn_test.py?")
    raise

  return (pdf, folds_df)


def main() -> None:
  df = flim.load_and_add_all()
  nn_exploration(df)
  plt.show()


if __name__ == "__main__":
  main()
