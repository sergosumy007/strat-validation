# Changelog

## 2026-06-11
- README: "How to read the metrics" section — what each number tells you and when
  it should worry you (mirrors the guide that ships inside paid audit reports).
- `examples/quickstart.py`: runnable demo of the full open core (CPCV + PSR/DSR +
  perp costs) on synthetic data, no market data required.
- README: "Beyond this repo" — what the paid audit adds on top of the open core,
  with a 56/100 -> 88/100 rework case.

## 2026-06-04
- `perp_costs.py`: funding / margin / liquidation modeling for USDT perpetuals (+22 tests).
- CI: GitHub Actions, Python 3.9 / 3.11 / 3.12, live badge.

## 2026-06-01
- Initial release: `cpcv.py`, `deflated_sharpe.py`, `lookahead_audit.py` + test suites.
