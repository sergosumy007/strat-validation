<div align="center">

# strat-validation

**The institutional toolkit for catching overfit trading strategies.**

Combinatorial Purged Cross-Validation · Probabilistic & Deflated Sharpe Ratio · static look-ahead audit · perpetual-futures cost modeling.
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
| `perp_costs.py` | *On perps, what do funding and leverage really cost — and where does the position get liquidated?* |

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

## 4 · Perpetual futures cost modeling (`perp_costs.py`)

Crypto perpetuals carry costs a candle-level backtest usually ignores: **funding** paid every 8h
while a position is open, and **liquidation** — a leveraged position can be wiped out *before* its
stop-loss is hit. This module models both for isolated, linear USDT-margined contracts. Pure
standard library, zero dependencies.

```python
from perp_costs import liquidation_price, funding_pnl, net_perp_pnl

# Where does a 10x long get liquidated? (isolated margin, 0.5% maintenance)
print(liquidation_price(entry=100, side="LONG", leverage=10, mmr=0.005))   # 90.5

# Funding paid by a long over the settlements it was held through (rate > 0 => long pays):
print(funding_pnl("LONG", notional=1000, rates=[0.0001, 0.0001, -0.0002]))  # 0.0

# Full realized PnL including round-trip taker fees + funding:
print(net_perp_pnl("LONG", entry=100, exit_price=110, qty=1,
                   funding_rates=[0.0005, 0.0005], fee_rate=0.00055))
```

Also exposes `liquidation_hit` (did a bar reach the liquidation level), `maintenance_margin`, and
`position_notional` (size a position so its stop-loss equals a fixed risk amount). The liquidation
identity is `liq_long = entry · (1 − 1/L + mmr + fee)` (short is symmetric); funding accrues over the
settlement rates the position was held across.

---

## How to read the metrics

What each number tells you, and when it should worry you:

| Metric | What it tells you |
|---|---|
| **PSR** | Probability the TRUE Sharpe beats the benchmark given sample size, skew, kurtosis. 95%+ is strong; below 60% the edge is not established. |
| **DSR** | PSR after charging for every configuration you tried. The honest number: high Sharpe with low DSR means you found noise by searching hard. |
| **SR0 (DSR barrier)** | The Sharpe a pure-noise search of M trials would produce. Your Sharpe must clear this bar before it means anything. |
| **MinTRL** | Minimum track record (trades) to confirm the edge at 95% confidence. If it exceeds your sample, the verdict is "not enough data", not "edge". |
| **CPCV folds** | Independent out-of-sample paths with purging + embargo. The edge must be positive on most paths, not on one lucky history. |
| **Look-ahead verdict** | Static audit of the code for future-data leaks (`arr[i+1]`, `shift(-1)`, unlagged rolling stats). One FAIL invalidates every other number. |
| **Liquidation price** | On perps a leveraged position can be wiped out before its stop-loss is hit. If your backtest never checks this, its drawdowns are fiction. |

Try it end-to-end on synthetic data:

```bash
python examples/quickstart.py
```

---

## Tests

```bash
pytest -q          # cpcv, deflated_sharpe, lookahead_audit and perp_costs are covered by SDET suites
```

Each module ships with self-tests against known-answer cases (e.g. purging is asserted to drop the
exact overlapping trades; PSR/DSR checked against reference values).

---

## License

MIT — use it, fork it, ship it.

---

## Beyond this repo

The four modules above are the open core. The paid audit runs a larger pipeline on top of them:

- **Bagged CPCV:** ensemble splits + stationary block bootstrap. The edge comes back as a
  confidence band, not a single number that may be luck.
- **Real-exit lens:** validation on the strategy's own exit logic, with transaction costs inside
  the engine. Fixed-RR proxies routinely misprice timeout and convex exits; I saw strategies flip
  from "no edge" to significant once measured on their true exit.
- **Crisis replay:** COVID, LUNA and FTX windows, not just the calm years.
- **Tail and capacity:** EVT/GPD tail fit, CVaR, and a square-root market-impact estimate of how
  much size the edge holds before slippage eats it.
- **Regime maps:** HMM and statistical jump models. The jump model switches state far less often,
  which is what you want from a kill-switch.
- **Universe honesty:** correlation clustering (d = √(2(1−ρ))) shows whether your 80 symbols are
  80 independent bets or five correlated ones.
- **Order-flow toxicity:** VPIN computed from plain OHLCV, no tick data required. Useful for
  entries that fade forced moves.

A recent internal case: a strategy scored 56/100 on its first validation pass. The failing tests
pointed at the causes (one side lost money everywhere, the edge only lived outside crisis regimes,
entries needed a toxicity filter). After those fixes the same pipeline scored it 88/100 with an
out-of-sample profit factor of 1.87. The score matters less than the diagnosis: the tests that fail
a strategy also tell you what to repair.

---

## Work with me

I run **fixed-price strategy validation audits** built on this exact stack — I tell you whether your
edge is real or just overfitting, in a clear bilingual report (executive verdict + charts + technical
appendix). If your backtest looks too good to be true, that's precisely what I check.

→ **[Hire me on Upwork](https://www.upwork.com/freelancers/~01e9b0869b70f8f855)**
