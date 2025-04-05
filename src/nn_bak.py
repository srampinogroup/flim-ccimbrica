#!/usr/bin/env python3
"""
This module tests neural networks on the same tabular dataset as the
module ``lr_test``. It is the only module needing Keras. K-fold
cross-validation is performed to compute the R² score.
"""
import time
import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
# Remove next two lines to use default tensorflow backend
import jax # for requirements.txt
os.environ["KERAS_BACKEND"] = "jax"
import keras
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout

import flim


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
  
  x = df[xlbl]
  # std = StandardScaler()
  # x = std.fit_transform(x)
  y = df[ylbl]
  x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                      test_size=flim.TEST_SIZE,
                                                      random_state=flim.RANDOM_STATE)
  r2s = []

  augment_props = np.linspace(0, 2, 4)

  for prop in augment_props:
    aug_df = pd.DataFrame(x_train, columns=xlbl)
    aug_df[ylbl] = y_train
    aug_df = flim.augment_dataset(aug_df, prop)
    x = aug_df[xlbl].values
    y = aug_df[ylbl]

    n_epochs = 800
    batch_size = 8

    # n_neurons = [70, 80, 90]
    n_neuron = 70
    # dropouts = [0, 0.03, 0.1, 0.3]
    dropout = 0

  # for n_neuron in n_neurons:
  # for dropout in dropouts:
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1],)))
    model.add(Dense(n_neuron, activation="relu"))
    model.add(Dropout(dropout))
    # model.add(Dense(n_neuron, activation="relu"))
    # model.add(Dropout(dropout))
    model.add(Dense(1, activation="relu")) # target is always > 0

    # TODO try mean_squared_logarithmic_error
    model.compile(loss="mean_squared_error",
                  optimizer="adamW",
                  metrics=["r2_score"])
    model.summary()

    n_splits = 2
    kfold = KFold(n_splits=n_splits, shuffle=True,
                  random_state=flim.RANDOM_STATE)
    loss = 0
    r2 = 0

    for i, (i_train, i_val) in enumerate(kfold.split(x_train,
                                                      y_train)):
      flim.log(f"Split {i} / {n_splits}")
      x_train = x[i_train, :]
      y_train = y.iloc[i_train]
      x_val = x[i_val, :]
      y_val = y.iloc[i_val]
      hist = model.fit(x_train, y_train,
                       epochs=n_epochs, batch_size=batch_size,
                       validation_data=(x_val, y_val), verbose=2)

      loss += model.evaluate(x_test, y_test, verbose=0)[0] \
              / n_splits
      y_pred = model.predict(x_test)
      r2 += r2_score(y_test, y_pred) / n_splits

    flim.log(f"{n_splits}-fold R², loss")
    print(r2, loss)
    r2s += [r2]

    _fig, _ax = plt.subplots()
    # plt.plot(hist.history["loss"])
    # plt.plot(hist.history["val_loss"])
    plt.plot(hist.history["r2_score"])
    plt.plot(hist.history["val_r2_score"])
    plt.xlabel("Epoch")
    plt.ylabel("$R^2$")
    plt.legend(["Train", "Test"])
    plt.title(f"{n_neuron} neurons, {dropout} dropout, {prop:.2f} "
              f"augmentation: $R² = {r2:.4f}$")

  _fig, _ax = plt.subplots()
  # plt.plot(n_neurons, r2s, marker="o")
  # plt.xlabel("number of neurons")
  # plt.plot(dropouts, r2s, marker="o")
  # plt.xlabel("dropout")
  plt.plot(augment_props, r2s, marker="o")
  plt.xlabel("augmentation proportion")
  plt.ylabel("$R^2$")

  y_pred = model.predict(x_test)
  pdf = pd.DataFrame({"y_test": y_test.values,
                      "y_pred": y_pred[:, 0].tolist()})
  print(pdf.sample(10))
  flim.log("R²")
  print(r2_score(pdf["y_test"], pdf["y_pred"]))

  _fig, _ax = plt.subplots()
  plt.scatter(y_test, y_pred)
  # plt.plot([0, 1], [0, 1], transform=_ax.transAxes, color="red")
  _ax.axline((1, 1), slope=1, color="red")
  plt.xlabel(f"real {ylbl} (units)")
  plt.ylabel(f"predicted {ylbl} (units)")

  # xi = range(len(y_test))
  # xs = sorted(xi, key=y_test.values.__getitem__)
  # _fig, _ax = plt.subplots()
  # plt.plot(xi, [y_test.iloc[i] for i in xs], marker="o",
  #     color="k", label=f"real ({ylbl})")
  # plt.plot(xi, [y_pred[i, 0] for i in xs], marker=".")
  # plt.xlabel("Sample ID (sorted by dosage")
  # plt.ylabel(f"Dosage ({flim.UNITS[ylbl]})")
  flim.log(f"Done in {time.process_time() - t0:.3f}.")


def main() -> None:
  df = flim.load_and_add_all()
  nn_exploration(df)
  plt.show()


if __name__ == "__main__":
  main()
