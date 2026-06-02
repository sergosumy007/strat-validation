"""
test_deflated_sharpe.py — self-tests (SDET) for deflated_sharpe.py.

Run:  python -m pytest test_deflated_sharpe.py -q
"""

import numpy as np
import pytest
from scipy import stats

from deflated_sharpe import (
    sharpe_ratio,
    probabilistic_sharpe_ratio,
    expected_max_sharpe,
    deflated_sharpe_ratio,
    min_track_record_length,
    EULER_MASCHERONI,
)


# ------------------------------------------------------- sharpe ratio ----
def test_sharpe_ratio_basic():
    r = np.array([1.0, 2.0, 3.0])  # mean=2, std(ddof=1)=1
    assert sharpe_ratio(r) == pytest.approx(2.0)


def test_zero_variance_raises():
    with pytest.raises(ValueError):
        sharpe_ratio(np.ones(50))


# --------------------------------------------------------------- PSR ----
def test_psr_half_when_sr_equals_benchmark():
    # numerator zero -> CDF(0) = 0.5, regardless of moments
    psr = probabilistic_sharpe_ratio(
        sr_observed=0.3, n=100, skew=0.0, kurtosis=3.0, sr_benchmark=0.3
    )
    assert psr == pytest.approx(0.5, abs=1e-9)


def test_psr_in_unit_interval():
    rng = np.random.default_rng(0)
    r = rng.normal(0.05, 1.0, size=500)
    psr = probabilistic_sharpe_ratio(r)
    assert 0.0 <= psr <= 1.0


def test_psr_increases_with_sample_size():
    # Same SR above benchmark -> more data gives higher confidence.
    common = dict(sr_observed=0.2, skew=0.0, kurtosis=3.0, sr_benchmark=0.0)
    p_small = probabilistic_sharpe_ratio(n=50, **common)
    p_large = probabilistic_sharpe_ratio(n=5000, **common)
    assert p_large > p_small > 0.5


def test_psr_increases_with_sr():
    common = dict(n=500, skew=0.0, kurtosis=3.0, sr_benchmark=0.0)
    assert (probabilistic_sharpe_ratio(sr_observed=0.4, **common)
            > probabilistic_sharpe_ratio(sr_observed=0.1, **common))


def test_psr_matches_manual_normal_formula():
    sr, n = 0.25, 250
    psr = probabilistic_sharpe_ratio(
        sr_observed=sr, n=n, skew=0.0, kurtosis=3.0, sr_benchmark=0.0
    )
    # normal case: denom_var = 1 + 0.5*sr^2
    denom = np.sqrt(1.0 + 0.5 * sr ** 2)
    z = sr * np.sqrt(n - 1) / denom
    assert psr == pytest.approx(float(stats.norm.cdf(z)), abs=1e-12)


def test_psr_requires_inputs():
    with pytest.raises(ValueError):
        probabilistic_sharpe_ratio(sr_observed=0.2)  # missing n/skew/kurtosis


# ----------------------------------------------- expected max sharpe ----
def test_expected_max_sharpe_single_trial_is_zero():
    assert expected_max_sharpe(variance_sr=0.5, n_trials=1) == 0.0


def test_expected_max_sharpe_matches_manual():
    V, M = 0.04, 10
    q1 = stats.norm.ppf(1.0 - 1.0 / M)
    q2 = stats.norm.ppf(1.0 - 1.0 / (M * np.e))
    expected = np.sqrt(V) * ((1 - EULER_MASCHERONI) * q1 + EULER_MASCHERONI * q2)
    assert expected_max_sharpe(V, M) == pytest.approx(expected, rel=1e-12)


def test_expected_max_sharpe_grows_with_trials():
    a = expected_max_sharpe(0.04, 5)
    b = expected_max_sharpe(0.04, 500)
    assert b > a > 0.0  # more trials -> higher inflation barrier


# --------------------------------------------------------------- DSR ----
def test_dsr_not_greater_than_psr_zero():
    # Deflation raises the bar, so DSR <= PSR(vs 0) whenever M > 1.
    rng = np.random.default_rng(1)
    r = rng.normal(0.06, 1.0, size=1000)
    out = deflated_sharpe_ratio(r, variance_sr=0.02, n_trials=50)
    assert out["dsr"] <= out["psr_vs_zero"] + 1e-12
    assert 0.0 <= out["dsr"] <= 1.0
    assert out["sr0"] > 0.0


def test_dsr_from_trial_list_derives_v_and_m():
    rng = np.random.default_rng(2)
    r = rng.normal(0.08, 1.0, size=2000)
    trials = [0.05, 0.08, 0.02, 0.11, 0.06, -0.01]
    out = deflated_sharpe_ratio(r, sr_trials=trials)
    assert out["n_trials"] == len(trials)
    # sr0 should equal expected_max_sharpe(var(trials), len(trials))
    V = np.var(np.asarray(trials, dtype=float), ddof=1)
    assert out["sr0"] == pytest.approx(expected_max_sharpe(V, len(trials)))


def test_dsr_single_trial_equals_psr_zero():
    rng = np.random.default_rng(3)
    r = rng.normal(0.05, 1.0, size=800)
    out = deflated_sharpe_ratio(r, variance_sr=0.03, n_trials=1)
    assert out["sr0"] == 0.0
    assert out["dsr"] == pytest.approx(out["psr_vs_zero"])


def test_dsr_requires_selection_inputs():
    rng = np.random.default_rng(4)
    r = rng.normal(0.05, 1.0, size=200)
    with pytest.raises(ValueError):
        deflated_sharpe_ratio(r)  # no sr_trials and no variance_sr/n_trials


# --------------------------------------------------- min track record ----
def test_min_trl_positive_and_requires_edge():
    trl = min_track_record_length(
        sr_observed=0.2, skew=0.0, kurtosis=3.0, sr_benchmark=0.0, confidence=0.95
    )
    assert trl > 1.0
    with pytest.raises(ValueError):
        # sr below benchmark -> infinite/undefined track record
        min_track_record_length(
            sr_observed=0.0, skew=0.0, kurtosis=3.0, sr_benchmark=0.1
        )
