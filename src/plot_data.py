#!/usr/bin/env python3
"""
This module is used to do most of the plots for the project. Some are
also done by ``lr_test`` and ``nn_test``, and the exponential fit is
in module ``fit_decay``.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import flim
import lr_test
import nn_test
import cnn
from plot_util import slug_figure_name, setup_defaults
from plot_util import OUT_PATH, FIG_WIDTH, FIG_HEIGHT


def plot_feature_importances(df: pd.DataFrame) -> None:
  """
  Plot forest estimator feature importance from FLIM dataset.
  """
  x_train, _xt, y_train, _yt = train_test_split(
      df[flim.FEATURES], df["dosage"],
      test_size=flim.TEST_SIZE,
      random_state=flim.RANDOM_STATE)
  model = flim.MODELS["For"]
  model.fit(x_train, y_train)
  _plot_fi_model(model, flim.FEATURES)


def _plot_fi_model(model, names: list[str]) -> None:
  """
  Plot a bar plot depicting importance of features in provided
  model. ``model`` must have ``feature_importances_`` and
  ``estimators_`` properties.
  """
  fi = model.feature_importances_
  sorted_indices = np.argsort(fi)
  names = [names[i] for i in sorted_indices]
  fi = [fi[i] for i in sorted_indices]
  _fig, _ax = plt.subplots()
  plt.barh(names, fi)
  flim.log(f"Saving to {OUT_PATH}/fig-fi.pdf...")
  plt.savefig(f"{OUT_PATH}/fig-fi.pdf")


def plot_r2_fixed(df: pd.DataFrame, fixed_label: str) -> None:
  """
  Generate target-label-dependant R² score for provided label,
  that is ``exposure`` or ``concentration``.'
  """
  if fixed_label == "exposure":
    ylbl = "concentration"
  else:
    ylbl = "exposure"

  exp_or_con = df[fixed_label].unique()
  r2_means = {}

  for name in flim.MODELS:
    r2_means[name] = [0] * len(exp_or_con)

  for _i in range(flim.N_REPEATS):
    for i, ue in enumerate(exp_or_con):
      flim.log(f"Fixed {fixed_label} = {ue}")
      udf = df[df[fixed_label] == ue]

      x_train, x_test, y_train, y_test = train_test_split(
          udf[flim.FEATURES], udf[ylbl],
          test_size=flim.TEST_SIZE,
          random_state=flim.RANDOM_STATE)

      for name, model in flim.MODELS.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        r2 = r2_score(y_test, y_pred)
        r2_means[name][i] += r2 / flim.N_REPEATS
        print(f"{name}: R² = {r2}$")

  _fig, _ax = plt.subplots()

  for name in flim.MODELS:
    plt.plot(exp_or_con, r2_means[name], label=f"$\\mathtt{{{name}}}$")

  plt.axhline(y=0, color="k", linestyle="--")
  plt.xlabel(f"Fixed {fixed_label} ({flim.UNITS[fixed_label]})")
  plt.ylabel("$R^2$")
  plt.xticks(exp_or_con)
  plt.legend()

  flim.log(f"Saving to {OUT_PATH}/fig-r2-{fixed_label}.pdf...")
  plt.savefig(f"{OUT_PATH}/fig-r2-{fixed_label}.pdf")


def plot_lr_test() -> None:
  """
  Plot ``lr_test`` results, that is R² scores for each model on
  different tests (fixed concentration, dosage, etc).
  """
  lr_results = lr_test.load_results()

  for test_name, test in lr_results.items():
    flim.log(f"Plotting {test_name}...")
    y_lbl = test["y_lbl"]
    x_test = test["x_test"]
    y_test = test["y_test"]

    xi = range(len(y_test))
    xs = sorted(xi, key=y_test.iloc.__getitem__)

    fig, _ax = plt.subplots()

    if test_name == "dosage":
      fig.set_size_inches((FIG_WIDTH * 2, FIG_HEIGHT * 1.5))

    plt.plot(xi, [y_test.iloc[i] for i in xs],
             color="k", label="real")

    for name, y_pred in test["y_pred"].items():
      r2 = r2_score(y_test, y_pred)
      plt.plot(xi, [y_pred[i] for i in xs],
               label=f"$\\mathtt{{{name}}}, R^2 = {r2:0.4f}$")

    reindexed = x_test.reset_index().reindex(index=xs).sort_index()
    plt.xticks(reindexed.index.values.tolist(),
               reindexed["index"].values.tolist(),
               rotation=90, fontsize=6)
    plt.xlabel(f"sample ID (sorted by {y_lbl})")
    units = flim.UNITS.get(y_lbl, "arbitrary units")
    plt.ylabel(f"{y_lbl} ({units})")
    plt.title(test_name)
    # plt.legend(loc="upper left")
    plt.legend()

    fig_name = slug_figure_name(test_name)
    flim.log(f"Saving to {OUT_PATH}/fig-{fig_name}.pdf...")
    plt.savefig(f"{OUT_PATH}/fig-{fig_name}.pdf")


def _plot_real_v_predicted(pdf, label: str, figfn: str) -> None:
  """
  Plot real versus predicted data. The ``figfn`` string is used as a
  fragment of the file name to avoid collision.
  """
  test = pdf["y_test"]
  pred = pdf["y_pred"]
  fig, ax = plt.subplots()
  fig.set_size_inches((FIG_WIDTH, FIG_HEIGHT * 0.6))
  ax.scatter(test, pred, marker=".")
  ax.axline((1, 1), slope=1, color="red")
  units = flim.UNITS[label]
  ax.set_xlabel(f"real {label} ({units})")
  ax.set_ylabel(f"predicted ({units})")
  ax.set_title(f"Test $R^2 = {r2_score(test, pred):.4f}$")
  out_path = f"{OUT_PATH}/fig-{figfn}-rp.pdf"
  flim.log(f"Saving to {out_path}...")
  plt.savefig(out_path)


def _plot_samples_pred(pdf: pd.DataFrame, label: str, figfn: str) -> None:
  """
  Plot predicted values over sorted real values. The ``figfn`` string
  is used as a fragment of the file name to avoid collision.
  """
  test = pdf["y_test"]
  pred = pdf["y_pred"]
  fig, ax = plt.subplots()
  fig.set_size_inches((FIG_WIDTH, FIG_HEIGHT * 0.6))
  ix = range(len(pdf.index))
  ax.plot(ix, test, marker=".", color="k")
  ax.plot(ix, pred, marker=".")
  # ax.set_xticks(ix, pdf.index.values, rotation=90, fontsize=6)
  ax.set_xticks([])
  ax.set_xlabel(f"sample (sorted by {label})")
  ax.set_ylabel(f"{label} ({flim.UNITS[label]})")
  # ax.set_title(f"Test $R^2 = {r2_score(test, pred):.4f}$")
  ax.legend(["test data", "prediction"])
  out_path = f"{OUT_PATH}/fig-{figfn}-pred.pdf"
  flim.log(f"Saving to {out_path}...")
  plt.savefig(out_path)


def _plot_fold_convergence(folds_df: pd.DataFrame, figfn: str) -> None:
  """
  Plot fold convergence of K-fold from ``nn_test`` or
  ``cnn_explore``. The ``figfn`` string
  is used as a fragment of the file name to avoid collision.
  """
  df = folds_df[folds_df["fold"] == 0]
  test_r2 = df["test_r2_score"].unique()[0]

  _fig, _ax = plt.subplots()
  plt.plot(df["r2_score"], marker="None")
  plt.plot(df["val_r2_score"], marker="None")
  plt.xlabel("Epoch")
  plt.ylabel("$R^2$")
  plt.legend(["train", "validation"])
  plt.title(f"Test $R^2 = {test_r2:.4f}$")

  out_path = f"{OUT_PATH}/fig-{figfn}-train.pdf"
  flim.log(f"Saving to {out_path}...")
  plt.savefig(out_path)


def plot_nn() -> None:
  """
  Plot figures associated with neural networks results.
  """
  pdf, folds_df = nn_test.load_results()
  _plot_real_v_predicted(pdf, "dosage", "nn")
  _plot_samples_pred(pdf, "dosage", "nn")
  _plot_fold_convergence(folds_df, "nn")


def plot_cnn() -> None:
  """
  Plot figures associated with convolutional neural networks results.
  """
  pdf, folds_df = nn_test.load_results()
  _plot_real_v_predicted(pdf, "dosage", "cnn")
  _plot_samples_pred(pdf, "dosage", "cnn")
  _plot_fold_convergence(folds_df, "cnn")


def main() -> None:
  """
  Plot R² scores for fixed exposure and concentration, feature
  importances and the results of ``lr_test``, ``nn_test`` and
  ``cnn_explore``.
  """
  setup_defaults()
  # df = flim.load_and_add_all()
  # plot_r2_fixed(df, "exposure")
  # plot_r2_fixed(df, "concentration")
  # plot_feature_importances(df)
  # plot_lr_test()
  plot_nn()
  plot_cnn()
  flim.log("Done.")
  # plt.show()


if __name__ == "__main__":
  main()
