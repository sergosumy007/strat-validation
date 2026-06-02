"""
deflated_sharpe.py — Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR).

Reference: David H. Bailey & Marcos Lopez de Prado,
    "The Sharpe Ratio Efficient Frontier" (2012) and
    "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest
     Overfitting and Non-Normality" (2014).

All Sharpe ratios here are **non-annualised** (per observation), exactly as the
formulas require. For a rule-based strategy a natural return series is the vector
of per-trade R-multiples: SR_hat = mean(R) / std(R). PSR/DSR then answer:

    PSR : "What is the probability the true Sharpe exceeds a benchmark SR*,
           given the sample size, skewness and kurtosis of the returns?"
    DSR : "...after inflating the benchmark to account for the M independent
           configurations that were tried during strategy selection?"

A DSR comfortably above 0.95 is the institutional bar for 'this edge is unlikely
to be a false discovery'.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy import stats

__all__ = [
    "sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "min_track_record_length",
]

EULER_MASCHERONI = 0.5772156649015329


def _moments(returns: np.ndarray):
    """Return (sr_hat, n, skew, kurtosis_full) for a 1-D return series."""
    r = np.asarray(returns, dtype=np.float64).ravel()
    n = r.size
    if n < 2:
        raise ValueError("need at least 2 returns")
    mean = r.mean()
    std = r.std(ddof=1)
    if std == 0:
        raise ValueError("zero-variance returns: Sharpe ratio is undefined")
    sr = mean / std
    sk = float(stats.skew(r, bias=False))
    # full (non-excess) kurtosis: normal == 3.0
    ku = float(stats.kurtosis(r, fisher=False, bias=False))
    return sr, n, sk, ku


def sharpe_ratio(returns: Sequence) -> float:
    """Non-annualised sample Sharpe ratio = mean / std(ddof=1)."""
    sr, _, _, _ = _moments(returns)
    return sr


def probabilistic_sharpe_ratio(
    returns: Optional[Sequence] = None,
    *,
    sr_observed: Optional[float] = None,
    n: Optional[int] = None,
    skew: Optional[float] = None,
    kurtosis: Optional[float] = None,
    sr_benchmark: float = 0.0,
) -> float:
    """
    Probabilistic Sharpe Ratio: P(true SR > sr_benchmark).

    Provide either ``returns`` (the series) OR all of
    ``sr_observed, n, skew, kurtosis`` precomputed.

    Formula:
        PSR(SR*) = Z[ (SR_hat - SR*) * sqrt(n - 1)
                      / sqrt(1 - g3*SR_hat + (g4 - 1)/4 * SR_hat^2) ]
    where g3 is skewness, g4 is full (non-excess) kurtosis, Z the standard
    normal CDF. All Sharpe ratios are non-annualised.
    """
    if returns is not None:
        sr_observed, n, skew, kurtosis = _moments(returns)
    else:
        if None in (sr_observed, n, skew, kurtosis):
            raise ValueError(
                "provide returns, or all of sr_observed/n/skew/kurtosis"
            )

    denom_var = 1.0 - skew * sr_observed + ((kurtosis - 1.0) / 4.0) * sr_observed ** 2
    if denom_var <= 0:
        raise ValueError(
            f"non-positive estimation variance ({denom_var:.4f}); "
            "PSR undefined for these moments"
        )

    z = (sr_observed - sr_benchmark) * np.sqrt(n - 1) / np.sqrt(denom_var)
    return float(stats.norm.cdf(z))


def expected_max_sharpe(variance_sr: float, n_trials: int) -> float:
    """
    Deflation benchmark SR_0 = expected maximum Sharpe across N independent
    trials when the true Sharpe is zero (the inflation barrier).

        SR_0 = sqrt(V) * [ (1 - gamma) * Z^-1(1 - 1/M)
                           + gamma * Z^-1(1 - 1/(M*e)) ]

    where V is the variance of the trial Sharpe ratios, M = n_trials,
    gamma is the Euler-Mascheroni constant and Z^-1 is the normal quantile.
    """
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1")
    if variance_sr < 0:
        raise ValueError("variance_sr must be non-negative")
    if n_trials == 1:
        return 0.0  # no selection bias with a single trial

    gamma = EULER_MASCHERONI
    q1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    q2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(variance_sr) * ((1.0 - gamma) * q1 + gamma * q2))


def deflated_sharpe_ratio(
    returns: Optional[Sequence] = None,
    *,
    sr_observed: Optional[float] = None,
    n: Optional[int] = None,
    skew: Optional[float] = None,
    kurtosis: Optional[float] = None,
    sr_trials: Optional[Sequence] = None,
    variance_sr: Optional[float] = None,
    n_trials: Optional[int] = None,
) -> dict:
    """
    Deflated Sharpe Ratio = PSR evaluated against the inflated benchmark SR_0.

    Selection-bias inputs (one of):
        * ``sr_trials``: the Sharpe ratios of every configuration that was tried
          (V and M are derived from it), or
        * ``variance_sr`` and ``n_trials`` directly.

    Returns a dict: {'dsr', 'sr0', 'psr_vs_zero', 'sr_observed', 'n_trials'}.
    """
    if returns is not None:
        sr_observed, n, skew, kurtosis = _moments(returns)
    elif None in (sr_observed, n, skew, kurtosis):
        raise ValueError("provide returns, or all of sr_observed/n/skew/kurtosis")

    if sr_trials is not None:
        trials = np.asarray(sr_trials, dtype=np.float64).ravel()
        if trials.size < 1:
            raise ValueError("sr_trials is empty")
        n_trials = int(trials.size)
        variance_sr = float(trials.var(ddof=1)) if trials.size > 1 else 0.0
    elif variance_sr is None or n_trials is None:
        raise ValueError("provide sr_trials, or both variance_sr and n_trials")

    sr0 = expected_max_sharpe(variance_sr, n_trials)
    dsr = probabilistic_sharpe_ratio(
        sr_observed=sr_observed, n=n, skew=skew, kurtosis=kurtosis,
        sr_benchmark=sr0,
    )
    psr0 = probabilistic_sharpe_ratio(
        sr_observed=sr_observed, n=n, skew=skew, kurtosis=kurtosis,
        sr_benchmark=0.0,
    )
    return {
        "dsr": dsr,
        "sr0": sr0,
        "psr_vs_zero": psr0,
        "sr_observed": float(sr_observed),
        "n_trials": int(n_trials),
    }


def min_track_record_length(
    returns: Optional[Sequence] = None,
    *,
    sr_observed: Optional[float] = None,
    skew: Optional[float] = None,
    kurtosis: Optional[float] = None,
    sr_benchmark: float = 0.0,
    confidence: float = 0.95,
) -> float:
    """
    Minimum Track Record Length: the number of observations required for PSR to
    reach ``confidence`` that the true Sharpe exceeds ``sr_benchmark``.

        MinTRL = 1 + (1 - g3*SR + (g4-1)/4*SR^2) * ( Z^-1(conf) / (SR - SR*) )^2
    """
    if returns is not None:
        sr_observed, _, skew, kurtosis = _moments(returns)
    elif None in (sr_observed, skew, kurtosis):
        raise ValueError("provide returns, or all of sr_observed/skew/kurtosis")

    if sr_observed <= sr_benchmark:
        raise ValueError("sr_observed must exceed sr_benchmark for a finite MinTRL")

    denom_var = 1.0 - skew * sr_observed + ((kurtosis - 1.0) / 4.0) * sr_observed ** 2
    z = stats.norm.ppf(confidence)
    return float(1.0 + denom_var * (z / (sr_observed - sr_benchmark)) ** 2)
