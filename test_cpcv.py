"""
test_cpcv.py — self-tests (SDET) for cpcv.py.

Run:  python -m pytest test_cpcv.py -q
"""

import numpy as np
import pandas as pd
import pytest

from cpcv import (
    CombinatorialPurgedCV,
    cpcv_backtest_paths,
    combinatorial_path_map,
    n_cpcv_paths,
)


# ---------------------------------------------------------------- counts ----
def test_path_count_formula():
    # phi[N,k] = C(N-1, k-1).  N=6,k=2 -> C(5,1)=5.
    assert n_cpcv_paths(6, 2) == 5
    assert n_cpcv_paths(10, 2) == 9
    assert n_cpcv_paths(4, 1) == 1


def test_get_n_splits_is_binomial():
    cv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2)
    assert cv.get_n_splits() == 15  # C(6,2)


# ----------------------------------------------------------- path map ----
def test_path_map_shape_and_coverage():
    combos, path_map = combinatorial_path_map(6, 2)
    assert len(combos) == 15
    assert path_map.shape == (5, 6)
    # every slot filled
    assert (path_map >= 0).all()
    # every group is a test group in exactly phi=5 splits
    for g in range(6):
        assert np.count_nonzero(path_map[:, g] >= 0) == 5


def test_path_map_trivial_k1():
    combos, path_map = combinatorial_path_map(4, 1)
    assert path_map.shape == (1, 4)
    # with k=1 the single path takes group g from split g
    assert list(path_map[0]) == [0, 1, 2, 3]


# ----------------------------------------------------- splitter basics ----
def test_train_test_disjoint_and_sized():
    X = np.zeros((12, 1))
    pt = np.arange(12)
    et = np.arange(12)
    cv = CombinatorialPurgedCV(n_splits=4, n_test_splits=1, embargo=0, purge=True)
    n_seen = 0
    for tr, te in cv.split(X, pred_times=pt, eval_times=et):
        n_seen += 1
        assert set(tr).isdisjoint(set(te))     # no leakage of indices
        assert te.size == 3                      # 12/4 = 3 per test group
    assert n_seen == 4                           # C(4,1)


# ------------------------------------------------- purge known-answer ----
def test_purge_known_answer():
    # 8 samples, 4 groups -> [[0,1],[2,3],[4,5],[6,7]]. Test group 1 = {2,3}.
    # t_start=i, t_end=i+2 so each obs spans forward.
    X = np.zeros((8, 1))
    pt = np.arange(8)
    et = np.arange(8) + 2
    cv = CombinatorialPurgedCV(n_splits=4, n_test_splits=1, embargo=0, purge=True)
    splits = list(cv.split(X, pred_times=pt, eval_times=et))
    # second split tests group 1 = {2,3}
    tr, te = splits[1]
    assert list(te) == [2, 3]
    # test interval [T_start, T_end] = [2, 5]; overlap purges train {0,1,4,5}
    assert list(tr) == [6, 7]


def test_embargo_known_answer():
    # Instantaneous labels so purge does nothing; isolate embargo.
    X = np.zeros((8, 1))
    pt = np.arange(8)
    et = np.arange(8)
    cv = CombinatorialPurgedCV(n_splits=4, n_test_splits=1, embargo=2, purge=True)
    tr, te = list(cv.split(X, pred_times=pt, eval_times=et))[1]
    assert list(te) == [2, 3]
    # right edge pos = 3; embargo window (3, 5] removes train starts {4,5}
    assert list(tr) == [0, 1, 6, 7]


def test_no_purge_keeps_all_train():
    X = np.zeros((8, 1))
    cv = CombinatorialPurgedCV(n_splits=4, n_test_splits=1, embargo=0, purge=False)
    tr, te = list(cv.split(X))[1]
    assert list(te) == [2, 3]
    assert list(tr) == [0, 1, 4, 5, 6, 7]   # nothing purged


def test_purge_requires_times():
    cv = CombinatorialPurgedCV(n_splits=4, n_test_splits=1, purge=True)
    with pytest.raises(ValueError):
        list(cv.split(np.zeros((8, 1))))


# --------------------------------------------- datetime time handling ----
def test_datetime_times_supported():
    X = np.zeros((8, 1))
    base = pd.Timestamp("2022-01-01")
    pt = pd.Series([base + pd.Timedelta(hours=i) for i in range(8)])
    et = pt + pd.Timedelta(hours=2)
    cv = CombinatorialPurgedCV(n_splits=4, n_test_splits=1, embargo=0, purge=True)
    tr, te = list(cv.split(X, pred_times=pt, eval_times=et))[1]
    assert list(te) == [2, 3]
    assert list(tr) == [6, 7]   # same overlap structure as the integer case


# ----------------------------------------------- backtest path builder ----
def test_backtest_paths_count_and_shape():
    df = pd.DataFrame({
        "t_start": np.arange(30),
        "t_end": np.arange(30) + 1,
        "ret": np.ones(30),
    })
    out = cpcv_backtest_paths(df, n_splits=6, n_test_splits=2)
    assert out["n_paths"] == 5
    assert out["metrics"].shape[0] == 5
    # constant +1 returns: sharpe=0 (zero variance), total=30, no drawdown
    m = out["metrics"].iloc[0]
    assert m["n"] == 30
    assert m["total_ret"] == pytest.approx(30.0)
    assert m["sharpe"] == 0.0
    assert m["max_drawdown"] == 0.0


def test_backtest_paths_drawdown():
    df = pd.DataFrame({
        "t_start": [0, 1, 2],
        "t_end": [1, 2, 3],
        "ret": [1.0, -2.0, 1.0],
    })
    out = cpcv_backtest_paths(df, n_splits=3, n_test_splits=1)
    assert out["n_paths"] == 1
    m = out["metrics"].iloc[0]
    assert m["total_ret"] == pytest.approx(0.0)     # equity 1 -> -1 -> 0
    assert m["max_drawdown"] == pytest.approx(-2.0)  # trough at -1 vs peak 1


def test_backtest_paths_missing_ret_raises():
    df = pd.DataFrame({"t_start": [0, 1, 2], "t_end": [1, 2, 3]})
    with pytest.raises(ValueError):
        cpcv_backtest_paths(df, n_splits=3, n_test_splits=1)


# ----------------------------------------------------- param validation ----
def test_invalid_params_raise():
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_splits=1)
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_splits=4, n_test_splits=4)  # k must be < N
    with pytest.raises(ValueError):
        CombinatorialPurgedCV(n_splits=4, n_test_splits=2, embargo=-0.1)
    with pytest.raises(ValueError):
        n_cpcv_paths(4, 5)
