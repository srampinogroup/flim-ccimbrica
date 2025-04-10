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

  fig, _ax = plt.subplots()
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
  out = OUT_PATH + "/fig-fitexp.pdf"
  flim.log(f"Saving to {out}...")
  plt.savefig(out)
  flim.log("Done.")

  flim.log("Generating TOC figure...")
  lw = 1
  ft = 8

  fig.set_size_inches((1.6, 2))
  plt.plot(tt, tc_pred, color="black", marker="None", lw=lw)
  # plt.text(124, 15e4, "$f(x) = a \\exp(-b t) + c$", fontsize=ft)
  plt.text(0.7, 15e4, "fit", fontsize=ft)
  plt.xlabel(None)
  plt.ylabel(None)
  plt.xticks([], [])
  plt.yticks([], [])
  plt.legend().remove()

  tix = np.argmax(c)
  plt.vlines(t[tix], 0, c[tix], color="C1", lw=lw)
  plt.text(0.6, -1, "tix", color="C1", fontsize=ft)
  plt.hlines(c[tix], t[tix], 1, color="C3", lw=lw)
  plt.text(0.9, 2e5, "max", color="C3", fontsize=ft)
  avg = 5e4
  plt.hlines(avg, t[0], t[-1], color="C4", lw=lw)
  plt.text(1.2, avg * 1.12, "avg", color="C4", fontsize=ft)

  out = OUT_PATH + "/fig-fitexp_toc.png"
  flim.log(f"Saving to {out}...")
  plt.savefig(out)
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
  plt.show()


if __name__ == "__main__":
  main()
