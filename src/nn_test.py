#!/usr/bin/env python3
"""
This module tests neural networks on the same tabular dataset as the
module ``lr_test``. It is the only module needing Keras. K-fold
cross-validation is performed to compute the R² score.
"""
import os
from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
# from lr_test import LRTest
os.environ["KERAS_BACKEND"] = "jax"
import keras
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout

import flim


keras.utils.set_random_seed(flim.RANDOM_STATE)


@dataclass
class NNTest(dict):
  """
  Dataclass for automation of neural network tests with fields:
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


def nn_test(df: pd.DataFrame) -> None:
  """
  Build a neural network and test accuracy with K-fold
  cross-validation using same features as ``lr_test``.
  """
  xlbl = flim.FEATURES
  ylbl = "dosage"

  x = df[xlbl]
  y = df[ylbl]

  std = StandardScaler()
  x = std.fit_transform(x)

  x_train, x_test, y_train, y_test = train_test_split(
      x, y, test_size=flim.TEST_SIZE,
      random_state=flim.RANDOM_STATE)

  n_epochs = 500

  r2s = []
  n_neurons = [20, 50, 100]
  # n_neurons = [100]
  # n_neuron = 100
  # dropouts = [0.02, 0.025, 0.03, 0.035, 0.04]
  dropout = 0.03

  for n_neuron in n_neurons:
  # for dropout in dropouts:
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1],)))
    model.add(Dense(n_neuron, activation="relu"))
    model.add(Dropout(dropout))
    model.add(Dense(n_neurons, activation="elu"))
    model.add(Dropout(0.03))
    model.add(Dense(1, activation="relu")) # target is always > 0

    model.compile(loss="mean_squared_error",
                  optimizer="adamW",
                  metrics=["r2_score"])
    model.summary()

    n_splits = 2
    kfold = KFold(n_splits=n_splits, shuffle=True,
                  random_state=flim.RANDOM_STATE)
    loss = 0
    r2 = 0

    for i, (i_train, i_test) in enumerate(kfold.split(x, y)):
      flim.log(f"Split {i} / {n_splits}")
      x_train = x[i_train, :]
      y_train = y.iloc[i_train]
      x_test = x[i_test, :]
      y_test = y.iloc[i_test]
      hist = model.fit(x_train, y_train,
                       epochs=n_epochs, batch_size=8,
                       validation_data=(x_test, y_test), verbose=2)
      loss += model.evaluate(x_test, y_test, verbose=0)[0] \
              / n_splits
      y_pred = model.predict(x_test)
      r2 += r2_score(y_test, y_pred) / n_splits

    flim.log(f"{n_splits}-fold R², loss")
    print(r2, loss)
    r2s += [r2]

  _fig, _ax = plt.subplots()
  plt.plot(n_neurons, r2s, marker="o")
  # plt.plot(dropouts, r2s, marker="o")

  # _fig, _ax = plt.subplots()
  # plt.plot(hist.history["loss"])
  # plt.plot(hist.history["val_loss"])
  # plt.xlabel("Epoch")
  # plt.ylabel("Loss")
  # plt.legend(["Train loss", "Validation loss"])

  # y_pred = model.predict(x_test)
  # pdf = pd.DataFrame({"y_test": y_test.values,
  #                     "y_pred": y_pred[:, 0].tolist()})
  # print(pdf.sample(10))
  # flim.log("R²")
  # print(r2_score(pdf["y_test"], pdf["y_pred"]))

  # _fig, _ax = plt.subplots()
  # plt.scatter(y_test, y_pred)
  # # plt.plot([0, 1], [0, 1], transform=_ax.transAxes, color="red")
  # _ax.axline((1, 1), slope=1, color="red")
  # plt.xlabel("Concentration (real)")
  # plt.ylabel("Concentration (predicted)")

  # xi = range(len(y_test))
  # xs = sorted(xi, key=y_test.values.__getitem__)
  # _fig, _ax = plt.subplots()
  # plt.plot(xi, [y_test.iloc[i] for i in xs], marker="o",
  #     color="k", label=f"real ({ylbl})")
  # plt.plot(xi, [y_pred[i, 0] for i in xs], marker=".")
  # plt.xlabel("Sample ID (sorted by dosage")
  # plt.ylabel(f"Dosage ({flim.UNITS[ylbl]})")


def main() -> None:
  df = flim.load_and_add_all()
  nn_test(df)
  flim.log("Done.")
  plt.show()


if __name__ == "__main__":
  main()
