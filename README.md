<div align="center">

# strat-validation

**The institutional toolkit for catching overfit trading strategies.**

Combinatorial Purged Cross-Validation · Probabilistic & Deflated Sharpe Ratio · static look-ahead audit.
Pure Python (NumPy / SciPy), no heavyweight dependencies, fully tested.

![python](https://img.shields.io/badge/python-3.9%2B-blue)
[![tests](https://github.com/sergosumy007/strat-validation/actions/workflows/tests.yml/badge.svg)](https://github.com/sergosumy007/strat-validation/actions/workflows/tests.yml)
![license](https://img.shields.io/badge/license-MIT-black)
![deps](https://img.shields.io/badge/deps-numpy%20%7C%20scipy%20%7C%20pandas-lightgrey)

</div>

---

## Why this exists

Most backtests look profitable for the wrong reasons:

1. **Selection bias** — the strategy was tuned on the same data it was tested on, so a good-looking
   Sharpe is just the maximum of many noisy trials.
2. **Look-ahead leakage** — the code quietly reads information that wouldn't have been available in
   real time (`arr[i+1]`, `shift(-1)`, a rolling stat computed without a lag…).

This toolkit attacks both, using the methods institutional desks actually rely on
(López de Prado, *Advances in Financial Machine Learning*, 2018; Bailey & López de Prado, 2014):

| Module | Question it answers |
|---|---|
| `cpcv.py` | *Does the edge survive many independent out-of-sample paths, not one lucky history?* |
| `deflated_sharpe.py` | *What's the probability the edge is real, after correcting for how many variants I tried?* |
| `lookahead_audit.py` | *Does my backtest code cheat by peeking into the future?* |

---

## Install

```bash
git clone https://github.com/sergosumy007/strat-validation.git
cd strat-validation
pip install -r requirements.txt   # numpy, scipy, pandas (+ pytest for tests)
pytest -q                         # run the test suite
```

---

## 1 · Combinatorial Purged Cross-Validation (`cpcv.py`)

An sklearn-style splitter that cuts the timeline into **N** groups, holds out every **k**-group
combination as test (`C(N, k)` splits → `C(N-1, k-1)` recombined backtest paths), and applies
**purging** (drop training trades whose evaluation interval overlaps the test block) and an
**embargo** window to kill leakage across the boundary.

```python
import numpy as np
from cpcv import CombinatorialPurgedCV, n_cpcv_paths

X = np.arange(1000)                       # any chronologically-sorted dataset
cv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2, embargo=0.01, purge=True)

print(cv.get_n_splits(), "train/test splits")   # C(6,2) = 15
print(n_cpcv_paths(6, 2), "recombined OOS paths")  # C(5,1) = 5

# pred_times = entry bar, eval_times = exit bar of each trade (drives purging)
for train_idx, test_idx in cv.split(X, pred_times=X, eval_times=X + 5):
    ...  # fit on train_idx, evaluate on test_idx
```

---

## 2 · Probabilistic & Deflated Sharpe Ratio (`deflated_sharpe.py`)

**PSR** — the probability the *true* Sharpe exceeds a benchmark, correcting for sample size, skew
and kurtosis. **DSR** — PSR measured against an inflated benchmark `SR₀` that accounts for the number
of trials `M` and the variance of their Sharpe ratios (the selection-bias barrier).

```python
import numpy as np
from deflated_sharpe import probabilistic_sharpe_ratio, deflated_sharpe_ratio

returns = np.random.default_rng(0).normal(0.05, 1.0, 500)   # per-trade R-multiples

psr = probabilistic_sharpe_ratio(returns, sr_benchmark=0.0)
print(f"PSR: {psr:.1%}")          # P(true Sharpe > 0)

# Deflate for the 40 configurations you actually tried:
res = deflated_sharpe_ratio(returns, variance_sr=0.02, n_trials=40)
print(f"DSR: {res['dsr']:.1%}  (barrier SR0={res['sr0']:.3f})")
# pass sr_trials=[...] instead to derive V and M from your trial Sharpes directly
```

**Formulas**

```
PSR(SR*) = Z[ (SR̂ − SR*)·√(n−1) / √(1 − γ₃·SR̂ + (γ₄−1)/4·SR̂²) ]
SR₀      = √V · [ (1−γ)·Z⁻¹(1 − 1/M) + γ·Z⁻¹(1 − 1/(M·e)) ]
DSR      = PSR(SR₀)
```

where `γ₃` skew, `γ₄` (non-excess) kurtosis, `γ` ≈ 0.5772 (Euler–Mascheroni), `Z` the normal CDF.
All Sharpe ratios are **non-annualised** (per-trade). `min_track_record_length()` is also included.

---

## 3 · Static look-ahead audit (`lookahead_audit.py`)

AST + regex scan of a strategy source file — catches the silent bugs that make a dead strategy look
alive. No execution, no imports of your code: it reads the file statically.

```python
from lookahead_audit import audit_lookahead

report = audit_lookahead("my_strategy.py")
print(report["verdict"])                       # PASS / WARN / FAIL
for f in report["findings"]:
    print(f["severity"], f["code"], f["title"])
```

Checks include forward indexing `arr[i+k]` inside `enumerate` loops, `shift(-N)`, `.at[i+N]`,
symmetric swing windows `[i-w:i+w]`, rolling stats without a lag, `close[-1]` references, a missing
transaction-cost model, and lagging-indicator misuse. Verdict severity is `FAIL > WARN > PASS`.

---

## Tests

```bash
pytest -q          # cpcv, deflated_sharpe and lookahead_audit are covered by SDET suites
```

Each module ships with self-tests against known-answer cases (e.g. purging is asserted to drop the
exact overlapping trades; PSR/DSR checked against reference values).

---

## License

MIT — use it, fork it, ship it.

---

## Work with me

I run **fixed-price strategy validation audits** built on this exact stack — I tell you whether your
edge is real or just overfitting, in a clear bilingual report (executive verdict + charts + technical
appendix). If your backtest looks too good to be true, that's precisely what I check.

→ **[Hire me on Upwork](https://www.upwork.com/freelancers/~01e9b0869b70f8f855)**
