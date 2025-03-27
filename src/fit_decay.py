#!/usr/bin/env python3
"""
Fit exponential decay to the decreasing part of the FLIM curve.

"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import flim
from plot_util import OUT_PATH, setup_defaults


def exp_decay(t, a, b, c) -> float:
  """
  Simple exponential decay with a, b, c being the amplitude, the
  decay rate (units of t⁻¹), and a constant term.

  """
  return a * np.exp(-b * t) + c


def fit_and_plot_decay(t: np.ndarray, c: np.ndarray) -> None:
  """
  Computes the fit and plot the decay curve.

  """
  tt, tc = flim.trunc_for_decay(t, c, 0.95)
  popt, pcov = curve_fit(exp_decay, tt, tc)

  _fig, _ax = plt.subplots()
  plt.scatter(t, c)
  plt.scatter(tt, tc, marker="x", lw=1)
  tc_pred = exp_decay(tt, *popt)
  plt.plot(tt, tc_pred, color="black", marker="None", lw=1)
  r2 = r2_score(tc, tc_pred)
  plt.xlim([0.3, 2])
  plt.xlabel("Time (ns)")
  plt.ylabel("Counts")
  plt.legend(["Counts of photons", "Truncated counts",
              f"Fit ($R^2$ = {r2:.4f})"])
  flim.log(f"Saving to {OUT_PATH}/fig-fitexp.pdf...")
  plt.savefig(f"{OUT_PATH}/fig-fitexp.pdf")
  plt.show()
  flim.log("Done.")


def main() -> None:
  """
  Load a random FLIM sample, fit the decay and plot the figure.

  """
  setup_defaults()
  df = flim.load_processed_flim()
  row = df.sample(1)
  t = row["time"].apply(np.array).iloc[0]
  c = row["counts"].apply(np.array).iloc[0]
  fit_and_plot_decay(t, c)


if __name__ == "__main__":
  main()
