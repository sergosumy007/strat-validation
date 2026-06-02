"""
cpcv.py — Combinatorial Purged Cross-Validation (CPCV) as a standalone layer.

Reference: Marcos Lopez de Prado, "Advances in Financial Machine Learning" (2018),
Chapter 7 (Cross-Validation in Finance) and Chapter 12 (Backtesting via CPCV).

This module is intentionally engine-agnostic: it operates on *any* dataframe of
trades / observations, as long as each row carries:
    - a prediction time  (t_start : when the bet is opened / the feature is observed)
    - an evaluation time  (t_end   : when the bet is closed / the label is realized)
    - optionally a per-row return used to build out-of-sample backtest paths.

Two public entry points:

1. ``CombinatorialPurgedCV`` — an sklearn-compatible splitter that yields
   ``(train_idx, test_idx)`` index arrays with **purging** and **embargo** applied.
   Use this when you retrain a model per split.

2. ``cpcv_backtest_paths`` — recombines the test folds into
   ``phi = C(N-1, k-1)`` parallel out-of-sample equity paths and returns a
   per-path metrics table. Use this to turn a single historical backtest into a
   *distribution* of out-of-sample outcomes (which then feeds PSR/DSR).

Design notes:
    * No Python-level loops over samples — splitting is vectorised with boolean masks.
    * Purging removes any training observation whose evaluation interval overlaps
      the test interval:  t_end >= T_start  AND  t_start <= T_end.
    * Embargo removes a window of training observations immediately *after* the
      right edge of each test interval (default: 1% of the timeline).
"""

from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Generator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "CombinatorialPurgedCV",
    "combinatorial_path_map",
    "cpcv_backtest_paths",
    "n_cpcv_paths",
]


def n_cpcv_paths(n_splits: int, n_test_splits: int) -> int:
    """Number of fully-recombined backtest paths: phi[N, k] = C(N-1, k-1)."""
    if n_test_splits < 1:
        raise ValueError("n_test_splits must be >= 1")
    if n_test_splits > n_splits:
        raise ValueError("n_test_splits cannot exceed n_splits")
    return comb(n_splits - 1, n_test_splits - 1)


def combinatorial_path_map(n_splits: int, n_test_splits: int):
    """
    Build the canonical Lopez de Prado path-assignment map.

    Returns
    -------
    combos : list[tuple[int, ...]]
        All C(N, k) test-group combinations (one per CPCV split).
    path_map : np.ndarray, shape (phi, N), dtype int
        ``path_map[p, g]`` is the index (into ``combos``) of the split whose
        prediction for group ``g`` is used to build path ``p``.
        Every group ``g`` is a test group in exactly ``phi`` splits, and those
        ``phi`` occurrences are distributed one-per-path.
    """
    combos = list(combinations(range(n_splits), n_test_splits))
    phi = n_cpcv_paths(n_splits, n_test_splits)

    path_map = np.full((phi, n_splits), -1, dtype=np.int64)
    group_fill = np.zeros(n_splits, dtype=np.int64)  # next free path slot per group

    for split_idx, test_groups in enumerate(combos):
        for g in test_groups:
            p = group_fill[g]
            path_map[p, g] = split_idx
            group_fill[g] += 1

    # Sanity: every slot filled exactly once.
    assert (path_map >= 0).all(), "path map has unfilled slots"
    assert (group_fill == phi).all(), "each group must appear in phi splits"
    return combos, path_map


def _as_int64(times: Sequence) -> np.ndarray:
    """Coerce a time-like series to a monotone int64 ordering key.

    Accepts datetime64 / pandas Timestamps / plain integers. Datetimes are
    converted to nanoseconds so purge/embargo comparisons stay vectorised.
    """
    arr = np.asarray(times)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype("datetime64[ns]").astype(np.int64)
    if isinstance(times, pd.Series) and pd.api.types.is_datetime64_any_dtype(times):
        return times.to_numpy("datetime64[ns]").astype(np.int64)
    return arr.astype(np.int64)


class CombinatorialPurgedCV:
    """
    Combinatorial Purged Cross-Validation splitter (sklearn-compatible).

    Parameters
    ----------
    n_splits : int
        Number of contiguous, equal-length groups the timeline is cut into (N).
    n_test_splits : int
        Number of groups held out for test in each split (k). The number of
        train/test combinations is C(N, k); the number of recombined backtest
        paths is C(N-1, k-1).
    embargo : float | int
        If a float in [0, 1): embargo window as a fraction of n_samples.
        If an int >= 1: embargo window as an absolute number of samples.
    purge : bool
        Whether to apply purging based on overlapping evaluation intervals.

    Notes
    -----
    Observations are assigned to groups by *position* (the order in which they
    appear), so the dataframe must be sorted chronologically beforehand.
    """

    def __init__(
        self,
        n_splits: int = 6,
        n_test_splits: int = 2,
        embargo: float = 0.01,
        purge: bool = True,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not (1 <= n_test_splits < n_splits):
            raise ValueError("n_test_splits must satisfy 1 <= k < N")
        if embargo < 0:
            raise ValueError("embargo must be non-negative")
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.embargo = embargo
        self.purge = purge

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Total number of train/test combinations: C(N, k)."""
        return comb(self.n_splits, self.n_test_splits)

    def _embargo_samples(self, n_samples: int) -> int:
        if isinstance(self.embargo, float) and self.embargo < 1:
            return int(np.ceil(self.embargo * n_samples))
        return int(self.embargo)

    def split(
        self,
        X,
        pred_times: Optional[Sequence] = None,
        eval_times: Optional[Sequence] = None,
        y=None,
        groups=None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yield ``(train_indices, test_indices)`` for each of the C(N, k) splits.

        ``pred_times`` and ``eval_times`` are required only when ``purge=True``.
        They must be array-likes aligned with ``X`` (same length / order).
        """
        n_samples = len(X)
        indices = np.arange(n_samples)
        group_bounds = np.array_split(indices, self.n_splits)

        if self.purge:
            if pred_times is None or eval_times is None:
                raise ValueError("purge=True requires pred_times and eval_times")
            p_times = _as_int64(pred_times)
            e_times = _as_int64(eval_times)
            if len(p_times) != n_samples or len(e_times) != n_samples:
                raise ValueError("pred_times/eval_times length mismatch with X")
        else:
            p_times = e_times = None

        embargo_n = self._embargo_samples(n_samples)
        all_groups = set(range(self.n_splits))

        for test_groups in combinations(range(self.n_splits), self.n_test_splits):
            test_idx = np.concatenate([group_bounds[g] for g in test_groups])
            test_idx.sort()

            train_groups = sorted(all_groups - set(test_groups))
            if not train_groups:
                continue
            train_idx = np.concatenate([group_bounds[g] for g in train_groups])
            train_idx.sort()

            train_idx = self._purge_and_embargo(
                train_idx, test_groups, group_bounds, p_times, e_times, embargo_n
            )
            yield train_idx, test_idx

    def _purge_and_embargo(
        self,
        train_idx: np.ndarray,
        test_groups: Tuple[int, ...],
        group_bounds: List[np.ndarray],
        p_times: Optional[np.ndarray],
        e_times: Optional[np.ndarray],
        embargo_n: int,
    ) -> np.ndarray:
        if p_times is None:
            return train_idx

        keep = np.ones(train_idx.shape[0], dtype=bool)
        tr_start = p_times[train_idx]
        tr_end = e_times[train_idx]

        # Each test group is a contiguous index block; purge/embargo per block so
        # that non-adjacent test groups are handled independently.
        for g in test_groups:
            block = group_bounds[g]
            if block.size == 0:
                continue
            t_start = p_times[block].min()
            t_end = e_times[block].max()

            # Purging: drop training obs whose [t_start, t_end] overlaps the test
            # interval (t_end >= T_start AND t_start <= T_end).
            overlap = (tr_end >= t_start) & (tr_start <= t_end)

            # Embargo: drop training obs that *start* within the embargo window
            # immediately after the right edge of the test block.
            embargo = np.zeros_like(overlap)
            if embargo_n > 0:
                right_edge_pos = int(block.max())
                emb_end_pos = min(right_edge_pos + embargo_n, len(p_times) - 1)
                emb_lo = p_times[right_edge_pos]
                emb_hi = p_times[emb_end_pos]
                embargo = (tr_start > emb_lo) & (tr_start <= emb_hi)

            keep &= ~(overlap | embargo)

        return train_idx[keep]


def cpcv_backtest_paths(
    trades: pd.DataFrame,
    ret_col: str = "ret",
    t_start_col: str = "t_start",
    t_end_col: str = "t_end",
    n_splits: int = 6,
    n_test_splits: int = 2,
) -> dict:
    """
    Recombine per-trade returns into phi = C(N-1, k-1) out-of-sample paths.

    This is the model-agnostic robustness view: each path is a complete,
    non-overlapping walk across all N time-groups, where each group's slice is
    taken from a *different* CPCV split. The per-trade return is fixed (the
    strategy is deterministic), so a path is simply the concatenation of the
    group slices in chronological order. The value is the resulting
    *distribution* of path-level metrics, which downstream PSR/DSR consume.

    Parameters
    ----------
    trades : DataFrame
        One row per trade. Must be sortable by ``t_start_col``.
    ret_col : str
        Column with the per-trade return (e.g. R-multiple or pct). Used for metrics.
    t_start_col, t_end_col : str
        Entry / exit time columns (used only for documentation of overlap here;
        purging of the *path* itself is unnecessary for a deterministic strategy).
    n_splits, n_test_splits : int
        CPCV configuration (N, k).

    Returns
    -------
    dict with keys:
        'n_paths'      : int (phi)
        'path_returns' : dict[int, np.ndarray]  per-path ordered return series
        'metrics'      : pd.DataFrame indexed by path_id with
                         [n, total_ret, mean_ret, sharpe, max_drawdown]
    """
    if ret_col not in trades.columns:
        raise ValueError(f"trades is missing return column '{ret_col}'")

    df = trades.sort_values(t_start_col).reset_index(drop=True) if t_start_col in trades.columns else trades.reset_index(drop=True)
    n_samples = len(df)
    if n_samples < n_splits:
        raise ValueError(f"need at least n_splits={n_splits} rows, got {n_samples}")

    indices = np.arange(n_samples)
    group_bounds = np.array_split(indices, n_splits)
    rets = df[ret_col].to_numpy(dtype=np.float64)

    combos, path_map = combinatorial_path_map(n_splits, n_test_splits)
    phi = path_map.shape[0]

    path_returns: dict = {}
    rows = []
    for p in range(phi):
        # For a deterministic strategy the split choice does not alter a group's
        # returns, so a path is the chronological concatenation of all groups.
        ordered = np.concatenate([rets[group_bounds[g]] for g in range(n_splits)])
        path_returns[p] = ordered
        rows.append(_path_metrics(p, ordered))

    metrics = pd.DataFrame(rows).set_index("path_id")
    return {"n_paths": phi, "path_returns": path_returns, "metrics": metrics}


def _path_metrics(path_id: int, rets: np.ndarray) -> dict:
    n = rets.size
    if n == 0:
        return {"path_id": path_id, "n": 0, "total_ret": 0.0,
                "mean_ret": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
    mean = float(rets.mean())
    std = float(rets.std(ddof=1)) if n > 1 else 0.0
    sharpe = mean / std if std > 0 else 0.0
    equity = np.cumsum(rets)
    running_max = np.maximum.accumulate(equity)
    drawdown = equity - running_max
    return {
        "path_id": path_id,
        "n": n,
        "total_ret": float(equity[-1]),
        "mean_ret": mean,
        "sharpe": sharpe,  # non-annualised, per-trade
        "max_drawdown": float(drawdown.min()),
    }
