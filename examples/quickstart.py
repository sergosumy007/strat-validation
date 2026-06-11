"""Quickstart: run the full open-core validation stack on synthetic trades.

    python examples/quickstart.py

Generates two synthetic strategies (one overfit, one with a real edge),
then runs CPCV, PSR/DSR and the perp cost model on both. No market data needed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from cpcv import CombinatorialPurgedCV, n_cpcv_paths
from deflated_sharpe import probabilistic_sharpe_ratio, deflated_sharpe_ratio
from perp_costs import liquidation_price, net_perp_pnl

rng = np.random.default_rng(11)
N = 600

# A: "overfit" — strong first half (the part you optimised on), noise after.
overfit = np.concatenate([rng.normal(0.35, 1.0, N // 2), rng.normal(-0.05, 1.0, N // 2)])
# B: "real edge" — small but stationary positive drift.
real = rng.normal(0.12, 1.0, N)

for name, r in (("A (overfit)", overfit), ("B (real edge)", real)):
    psr = probabilistic_sharpe_ratio(r, sr_benchmark=0.0)
    dsr = deflated_sharpe_ratio(r, variance_sr=0.02, n_trials=40)

    cv = CombinatorialPurgedCV(n_splits=6, n_test_splits=2, embargo=0.01)
    idx = np.arange(N)
    fold_means = []
    for _, test_idx in cv.split(idx, pred_times=idx, eval_times=idx + 5):
        fold_means.append(r[test_idx].mean())
    fold_means = np.array(fold_means)

    print(f"\n=== {name} ===")
    print(f"mean R per trade : {r.mean():+.3f}")
    print(f"PSR              : {psr:.1%}")
    print(f"DSR (40 trials)  : {dsr['dsr']:.1%}   (barrier SR0={dsr['sr0']:.3f})")
    print(f"CPCV             : {cv.get_n_splits()} splits, {n_cpcv_paths(6, 2)} paths, "
          f"{(fold_means > 0).sum()}/{len(fold_means)} folds positive")

print("\n=== perp cost sanity ===")
print(f"10x long from 100 liquidates at {liquidation_price(100, 'LONG', 10):.2f}")
print(f"net PnL long 100->110, 2 funding ticks: "
      f"{net_perp_pnl('LONG', 100, 110, 1, funding_rates=[5e-4, 5e-4]):+.4f}")

print("\nTakeaway: A and B can show similar raw Sharpe, but DSR and the CPCV fold"
      " split separate the searched-for noise from the stationary edge.")
