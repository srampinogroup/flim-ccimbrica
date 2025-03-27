#!/usr/bin/env python3
"""
Provide helper functions for plotting. Only
``alternate_every_xtick`` and defaults for matplotlib.

"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator, FuncFormatter


FIG_WIDTH = 238 / 72
FIG_HEIGHT = FIG_WIDTH * 0.75
OUT_PATH = "plots"


def alternate_every_xtick(axes: Axes,
                          ticks: list[int | float],
                          labels: list[int | float | str],
                          pad: int = 12) -> None:
  """
  Set xticks on an axis suck that every label is displayed, but
  every other one is offset vertically to reduce overlap. Offset
  in points is set by the offset.
  Use in place of ``plt.xtick(ticks, labels)``.

  """
  def formatter(_x, p):  # pylint: disable=unused-argument
    return labels[p * 2 + 1]

  axes.set_xticks(ticks, labels)
  ax = axes.xaxis
  ax.set_major_locator(FixedLocator(ticks[::2]))
  ax.set_minor_locator(FixedLocator(ticks[1::2]))
  ax.set_minor_formatter(FuncFormatter(formatter))
  axes.tick_params(which="major", pad=pad, axis="x")


def slug_figure_name(name: str) -> str:
  """
  Escape figure name to be a valid file name.

  """
  fn = "".join(c for c in name if c.isalnum() or c in "_-")
  return fn.replace("µ", "u")


def setup_defaults() -> None:
  """
  Set default parameters for plotting.

  """
  plt.rcParams["figure.figsize"] = (FIG_WIDTH, FIG_HEIGHT)
  plt.rcParams['figure.constrained_layout.use'] = True
  plt.rcParams["font.size"] = 8
  plt.rcParams["axes.titlesize"] = 8
  plt.rcParams["lines.linewidth"] = 0.5
  plt.rcParams["lines.marker"] = "."
  plt.rcParams["lines.markersize"] = 3
  plt.rcParams["legend.labelspacing"] = 0
  plt.rcParams["legend.fontsize"] = 6
  plt.rcParams["legend.framealpha"] = 0.4
  Path("plots").mkdir(exist_ok=True)


def main() -> None:
  """
  Make a dummy plot as testing for ``alternate_every_xtick``.

  """
  n = 40
  x = range(50, 50 + n)
  y = list(map(lambda x: 1 / x, x))

  _fig, ax = plt.subplots()
  plt.plot(x, y)
  plt.xticks(x, x)

  _fig, ax = plt.subplots()
  plt.plot(x, y)
  alternate_every_xtick(ax, x, x)

  plt.show()


if __name__ == "__main__":
  main()
