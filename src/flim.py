#!/usr/bin/env python3
"""
FLIM project common functions. This module provides functions to
read and preprocess the data, and the models used for our analysis.
All computation uses the same ``RANDOM_STATE`` by seeding the numpy
library with it. Other computations that need to be done several
times for averaging are done ``N_REPEATS`` times. The estimators used
are accessible via the ``MODELS`` dictionary.

See ***DOI PAPER.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from colorama import init as _colorama_init
from colorama import Fore, Style
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model as lm


DEFAULT_DF_PATH = "flimdf.json"
RANDOM_STATE = 1
N_REPEATS = 10
TEST_SIZE = 0.1
MODELS = {
  "Lin": lm.LinearRegression(positive=True),
  "Rid": lm.Ridge(solver="svd", alpha=0.1),
  "For": RandomForestRegressor(
      criterion="absolute_error",
      n_estimators=500,
      max_depth=15,
      min_samples_split=20,
      min_samples_leaf=8,
      max_features=0.35),
  "GBR": GradientBoostingRegressor(
      loss="absolute_error",
      n_estimators=200,
      learning_rate=0.1),
}
FEATURES = [
  "counts_avg", "counts_std", "counts_skew", "counts_tix",
  "fit_rate", "fit_const",
  "counts_avg*fit_rate", "counts_skew*fit_const"
]
UNITS = {
  "concentration": "µg/ml",
  "exposure": "h",
  "dosage": "µg/ml h",
}


def log(s: str) -> None:
  """
  Print a message with a color in std out.
  """
  print(f"{Fore.BLUE}{Style.BRIGHT}{s}{Style.RESET_ALL}")


def err(s: str) -> None:
  """
  Error message in std err.
  """
  print(f"{Fore.RED}{Style.BRIGHT}{s}{Style.RESET_ALL}",
        file=sys.stderr)


def read_flim_df(path: str = DEFAULT_DF_PATH) -> pd.DataFrame:
  """
  Read the FLIM data from JSON file.
  """
  log("Reading the FLIM data...")
  init()
  # df = pd.read_json(path, dtype_backend="pyarrow")
  df = pd.read_json(path)
  df.columns = [
    "time",
    "counts",
    "date",
    "exposure",
    "concentration",
    "cell"
  ]
  log(f"Raw data has shape {df.shape}.")
  return df


def _filter_df(df: pd.DataFrame) -> pd.DataFrame:
  """
  Filter out the problematic batches.
  """
  log("Filtering...")

  good_mask = df["date"].isin([
    "08-02-21",
    "09-02-21",
    "10-02-21",
    "11-02-21",
    "12-02-21",
  ])

  # good_mask = df["date"] != "20-01-21"
  # good_mask &= df["date"] != "04-03-21"

  # dt = df["time"].apply(lambda ts: ts[1] - ts[0])
  # ds, ct = np.unique(dt, return_counts=True)
  # delta_t = ds[ct.argmax()]

  # print(f"Most common Delta t is {delta_t} with {ct.max()} rows.")
  # good_mask &= dt == delta_t

  # sz_before = len(df)
  df = df[good_mask]

  # print(f"Removed {sz_before - len(df)} rows total.")
  log(f"Filtered data has shape {df.shape}.")
  return df


def _uniform_counts_size(df: pd.DataFrame) -> pd.DataFrame:
  """
  Uniformize the time and counts size.
  """
  log("Uniforming the time and counts lengths...")
  sizes = df["counts"].apply(len)
  min_size = sizes.min()
  med_size = sizes.median()
  max_size = sizes.max()
  print("Counts:")
  print(f"min: {min_size}, median: {med_size}, max: {max_size}")

  if min_size * 2 < max_size:
    warnings.warn(f"{Fore.ORANGE}Dataframe probably contains "
                  "data with different Delta t.{Style.RESET_ALL}")

  print(f"Truncating time and counts length to {min_size}...")

  def trunc(x):
    return x[:min_size]

  df["time"] = df["time"].apply(trunc)
  df["counts"] = df["counts"].apply(trunc)

  assert (df["time"].apply(len) == min_size).all(), \
    "Time lengths are not the same."
  assert (df["counts"].apply(len) == min_size).all(), \
    "Counts lengths are not the same."

  return df


def load_processed_flim(path: str = DEFAULT_DF_PATH) -> pd.DataFrame:
  """
  Load the dataset, filter out bad batches and make sure every row
  has the same time and counts size.
  """
  df = read_flim_df(path)
  df = _filter_df(df)
  df = _uniform_counts_size(df)
  return df


def add_normalize_counts(df: pd.DataFrame) -> None:
  """
  Harmonize counts across cells into new features (in place).
  """
  def normalize_counts(c):
    m = np.max(c)
    return [x / m for x in c]

  log("Normalizing counts by max...")
  df["counts_norm"] = df["counts"].apply(normalize_counts)


def add_stat_features(df: pd.DataFrame) -> None:
  """
  Add statistics of counts as features (in place).
  """
  log("Adding stats features...")

  keys = ["counts"]
  if "counts_norm" in df.index:
    keys += ["counts_norm"]

  time = df["time"].iloc[0]

  for counts_key in keys:
    df[counts_key + "_max"] = df[counts_key].apply(np.max)
    df[counts_key + "_tix"] = df[counts_key].apply(
        lambda r: time[np.argmax(r)])
    df[counts_key + "_avg"] = df[counts_key].apply(np.mean)
    df[counts_key + "_std"] = df[counts_key].apply(np.std)
    df[counts_key + "_std-1"] = 1.0 / df[counts_key + "_std"]
    df[counts_key + "_skew"] = \
      (df[counts_key + "_avg"] - df[counts_key + "_max"]) \
      / df[counts_key + "_std"] * 3
    df[counts_key + "_skew-1"] = 1.0 / df[counts_key + "_skew"]


def trunc_for_decay(t: np.ndarray, c: np.ndarray, p: float = 0.95) \
    -> tuple[np.ndarray, np.ndarray]:
  """
  Truncate the data for exponential decay fitting by removing
  everything before the peak, and also a bit after the peak.
  t: time,
  c: counts,
  p: max peak fraction (i.e. 0.5 for half-max).
  """
  peak_ix = c.argmax()
  # cut the long tail to focus on the decay part
  last_ix = t.size // 2

  if p < 1:
    peak = c[peak_ix]
    peak_ix = np.where(c > peak * p)[0][-1]

  return t[peak_ix:last_ix], c[peak_ix:last_ix]


def exp_decay(t, a, b, c) -> float:
  """
  Simple exponential decay with a, b, c being the amplitude, the
  decay rate (units of t⁻¹), a constant term.
  """
  return a * np.exp(-b * t) + c


def add_decay_fit_features(df: pd.DataFrame) -> None:
  """
  Add exponential decay fit parameters (in place).
  """
  def exp_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

  def map_fit(row):
    time = np.array(row["time"])
    counts = np.array(row["counts"])
    t, c = trunc_for_decay(time, counts)
    popt, _ = curve_fit(exp_decay, t, c)
    return popt

  log("Adding fit parameters...")

  key_a = "fit_amp"
  key_b = "fit_rate"
  key_c = "fit_const"

  df[[key_a, key_b, key_c]] = df[["time", "counts"]].apply(
      map_fit, axis=1, result_type="expand")

  df[key_b + "-1"] = 1.0 / df[key_b]


def add_interaction_terms(df: pd.DataFrame) -> None:
  """
  Add interaction terms of different features that might be relevant
  for feature exploration (in place).
  """
  log("Adding interaction terms...")
  combi = [
    ("+", lambda a, b: a + b),
    ("*", lambda a, b: a * b),
    ("/", lambda a, b: a / b),
  ]

  features = ["counts_avg", "counts_std", "counts_skew",
              "fit_rate", "fit_const"]
  n = len(features)
  r = range(n)

  for i, j in ((i, j) for j in r for i in r if i < j):
    a = features[i]
    b = features[j]

    for op, lam in combi:
      df[a + op + b] = lam(df[a], df[b])


def add_conexp_labels(df: pd.DataFrame) -> None:
  """
  Add concentration-exposure combinations (in place).
  """
  log("Adding conexp labels...")
  df["dosage"] = df["concentration"] * df["exposure"]
  df["con.exp"] = ((df["concentration"] + 1) *
                   (df["exposure"] + 1))
  df["con_norm"] = df["concentration"] / df["concentration"].max()
  df["exp_norm"] = df["exposure"] / df["exposure"].max()
  df["dosage_norm"] = df["con_norm"] * df["exp_norm"]
  df["con+exp_norm"] = df["con_norm"] + df["exp_norm"]


def load_and_add_all() -> pd.DataFrame:
  """
  Load dataset and add all features, labels and interactions to have
  the comprehensive dataset for exploring the data.
  """
  df = load_processed_flim()
  add_normalize_counts(df)
  add_stat_features(df)
  add_decay_fit_features(df)
  add_interaction_terms(df)
  add_conexp_labels(df)
  return df.copy()


def augment_dataset(df: pd.DataFrame,
                     proportion: float = 1.0) -> pd.DataFrame:
  """
  Augment the dataset by adding random variation to existing samples
  to increase the number of samples. A proportion of 1.0 means
  doubling the dataset size.
  """
  def randomize(value: float, std: float) -> float:
    rnd = np.random.normal(0, std)
    return abs(value + rnd / 10)

  log("Augmenting dataset...")

  excluded = set(["counts", "counts_norm", "time", "date", "cell"])

  aug_df = df.sample(int(len(df) * proportion),
                     random_state=RANDOM_STATE).copy()
  for column in set(aug_df.keys()) - excluded:
    std = df[column].std()
    aug_df[column] = aug_df[column].apply(
        lambda row, std_=std: randomize(row, std_))

  return pd.concat([df, aug_df], ignore_index=True)



_is_init = False

def init() -> None:
  """
  Lazy-initialize colorama, matplotlib backend and set bigger
  margin for pandas printouts. Or call manually in interactive shell.
  Idempotent.

  """
  global _is_init
  if not _is_init:
    _colorama_init(autoreset=True)
    pd.set_option("display.max_rows", 500)
    pd.set_option("display.max_columns", 500)
    pd.set_option("display.width", 150)
    np.random.seed(RANDOM_STATE)
    _is_init = True


def export_sample_data(df: pd.DataFrame = None,
                       outpath: str = "out/sampledf.txt") -> None:
  """
  Exports 5 first and 5 lasts rows as a human readable sample of
  the passed FLIM dataframe ``df`` into ``outpath`` file.
  """
  if df is None:
    df = read_flim_df("flimdf.json")

  def firsts_as_str(c: list, n: int = 2) -> str:
    return f"[{str(c[:n])[1:-1]}...]"

  for i in ["counts", "time"]:
    df[i] = df[i].apply(firsts_as_str)

  log(f"Writing to {outpath}...")
  Path(outpath).parent.mkdir(parents=True,
                             exist_ok=True)
  with open(outpath, "w", encoding="ascii") as f:
    f.write(df.to_string(max_rows=10))
  log("Done.")


def main() -> None:
  """
  Simply export a subset of the data if called directly.
  """
  export_sample_data()


if __name__ == "__main__":
  main()
