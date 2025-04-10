#!/usr/bin/env python3
"""
Convolutional neural network on the full time-serie signal.
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
# Remove next two lines to use default tensorflow backend
import jax # for requirements.txt
os.environ["KERAS_BACKEND"] = "jax"
import keras
from keras import layers
from keras.callbacks import EarlyStopping

import flim
import plot_data


RESULTS_FILE = "out/cnn_results.json"
FOLDS_FILE = "out/cnn_folds.json"


keras.utils.set_random_seed(flim.RANDOM_STATE)


def cnn_explore(df: pd.DataFrame) -> None:
  """
  Exploration of CNN hyperparameters using basic linear search and
  keeping best.
  """
  t0 = time.process_time()
  n_epochs = 2000
  # batch_sizes = [20, 60, 80, 90]
  batch_size = 60
  # learning_rates = [0.0001, 0.001, 0.003, 0.005]
  # learning_rate = 0.003
  # weight_decays = np.linspace(0.001, 0.01, 10)
  # weight_decay = 0.001
  # augment_props = np.linspace(0, 1, 5)
  # prop = 0.0
  # n_neurons = [5, 10, 15]
  n_neuron = 5
  # n_filters = [3, 5, 7]
  n_filter = 3
  # kernel_sizes = [2, 3, 4, 5]
  kernel_size = 3
  # dropouts = np.insert(np.linspace(0.075, 0.15, 24), 0, 0)
  dropout = 0.1

  n_splits = 5
  optimizer = "adamW"
  # optimizer = keras.optimizers.AdamW(learning_rate,
  #                                    weight_decay=weight_decay)
  callbacks = [EarlyStopping(monitor="val_r2_score", patience=10)]

  r2s = []
  folds_df = pd.DataFrame(columns=["fold",
                                   "r2_score",
                                   "val_r2_score",
                                   "test_r2_score"])

  for _dummy in [1]:
  # for batch_size in batch_sizes:
  # for learning_rate in learning_rates:
  # for weight_decay in weight_decays:
  # for prop in augment_props:
  # for n_neuron in n_neurons:
  # for n_filter in n_filters:
  # for kernel_size in kernel_sizes:
  # for dropout in dropouts:

    nc = len(df["time"].iloc[0])
    # x = df["counts"]
    x = np.array([np.array(counts)
                  for counts in df["counts_norm"].values])
    x = x.reshape(-1, nc, 1)
    y = df["dosage"].values

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=flim.TEST_SIZE,
        random_state=flim.RANDOM_STATE)

    model = keras.Sequential([
      layers.Input(shape=(nc, 1)),
      layers.Conv1D(filters=n_filter, kernel_size=kernel_size, 
                    use_bias=True, activation="relu"),
      layers.MaxPooling1D(pool_size=2),
      # layers.GlobalAveragePooling1D(),
      # layers.Conv1D(filters=32, kernel_size=3, activation="relu"),
      # layers.MaxPooling1D(pool_size=2),
      layers.Flatten(),
      layers.Dense(n_neuron, activation="relu"),
      layers.Dense(1, activation="relu"),
    ])

    model.compile(loss="mean_squared_error",
                  metrics=["r2_score"],
                  optimizer=optimizer)
    model.summary()

    kfold = KFold(n_splits=n_splits, shuffle=True,
                  random_state=flim.RANDOM_STATE)
    loss = 0
    r2_tot = 0

    for i, (i_train, i_test) in enumerate(kfold.split(x, y)):
      flim.log(f"Split {i} / {n_splits}")
      x_train = x[i_train, :]
      y_train = y[i_train]
      x_val = x[i_test, :]
      y_val = y[i_test]
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
    r2s += [r2]

  # _fig, _ax = plt.subplots()
  # plt.plot(n_neurons, r2s, marker="o")
  # plt.xlabel("number of neurons")
  # plt.plot(n_filters, r2s, marker="o")
  # plt.xlabel("number of filters")
  # plt.plot(kernel_sizes, r2s, marker="o")
  # plt.xlabel("size of kernel")
  # plt.plot(dropouts, r2s, marker="o")
  # plt.xlabel("dropout")
  # plt.plot(augment_props, r2s, marker="o")
  # plt.xlabel("augmentation proportion")
  # plt.plot(batch_sizes, r2s, marker="o")
  # plt.xlabel("batch size")
  # plt.plot(learning_rates, r2s, marker="o")
  # plt.xlabel("learning rate")
  # plt.plot(weight_decays, r2s, marker="o")
  # plt.xlabel("weight decay")
  # plt.ylabel("$R^2$")

  y_pred = model.predict(x_test)
  pdf = pd.DataFrame({"y_test": y_test,
                      "y_pred": y_pred[:, 0].tolist()})
  pdf.sort_values(by="y_test", inplace=True)
  print(pdf.sample(10))
  flim.log("R²")
  print(r2_score(pdf["y_test"], pdf["y_pred"]))

  save_results(pdf, folds_df)

  flim.log(f"Done in {time.process_time() - t0:.3f} s.")


def save_results(pdf: pd.DataFrame, folds_df: pd.DataFrame) -> None:
  """
  Save the results of ``cnn_explore`` to JSON files. Files are
  overwritten. Paths may be created.
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
  Load the results of ``cnn_explore`` from JSON files.
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
             "did you run cnn.py?")
    raise

  return (pdf, folds_df)


def main() -> None:
  cnn_explore(flim.load_and_add_all())
  plt.show()


if __name__ == "__main__":
  main()
