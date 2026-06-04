"""
perp_costs.py — Realistic cost & risk modeling for crypto perpetual futures.

Standalone, dependency-light (stdlib + optional float math only). Models the
three things a candle-level backtest usually ignores on perps:

  1. Funding payments accrued while a position is held (every 8h on Bybit/Binance)
  2. Maintenance margin (isolated, linear USDT-margined contracts)
  3. Liquidation price implied by leverage — a leveraged position can be wiped
     out BEFORE its stop-loss is reached.

Conventions
-----------
* Linear USDT-margined perpetuals, ISOLATED margin, one-way mode.
* side: 'LONG' or 'SHORT' (case-insensitive).
* Funding rate sign: positive funding rate => longs pay shorts.
* All formulas are documented approximations of the venue engine (tier
  maintenance-amount deductions are ignored => slightly conservative liq price).

Formulas (isolated linear long, derived from equity == maintenance margin):
    IM/notional + (P/entry - 1) == mmr + fee_rate
    => liq_long  = entry * (1 - 1/L + mmr + fee_rate)
       liq_short = entry * (1 + 1/L - mmr - fee_rate)

These are the same identities used in the project backtest engine (Block 2)
and exposed here as a clean, testable module.
"""
from __future__ import annotations

from typing import Sequence

__all__ = [
    'position_notional',
    'maintenance_margin',
    'liquidation_price',
    'liquidation_hit',
    'funding_pnl',
    'net_perp_pnl',
]


def _side_sign(side: str) -> int:
    """+1 for LONG, -1 for SHORT."""
    s = side.strip().upper()
    if s in ('LONG', 'BUY', 'L'):
        return 1
    if s in ('SHORT', 'SELL', 'S'):
        return -1
    raise ValueError(f"side must be LONG or SHORT, got {side!r}")


def position_notional(risk_amount: float, entry: float, sl_price: float) -> float:
    """
    Notional (entry * qty) of a position sized so that hitting `sl_price`
    loses exactly `risk_amount` (the R-framework used by the engine).

    qty = risk_amount / |entry - sl_price|;  notional = qty * entry.
    """
    dist = abs(entry - sl_price)
    if dist <= 0 or entry <= 0:
        return 0.0
    qty = risk_amount / dist
    return qty * entry


def maintenance_margin(notional: float, mmr: float, maint_amount: float = 0.0) -> float:
    """
    Maintenance margin requirement for a position of given notional.
    mmr = maintenance margin rate (e.g. 0.005 = 0.5%). `maint_amount` is the
    tier deduction (default 0 => conservative).
    """
    return max(0.0, notional * mmr - maint_amount)


def liquidation_price(entry: float, side: str, leverage: float,
                      mmr: float = 0.005, fee_rate: float = 0.0) -> float:
    """
    Isolated-margin liquidation price for a linear perpetual.

    entry     : entry price
    side      : 'LONG' / 'SHORT'
    leverage  : position leverage (notional / initial_margin), > 0
    mmr       : maintenance margin rate (default 0.5%)
    fee_rate  : taker fee charged on close, widens the liq toward entry (default 0)

    Returns the price at which the position is liquidated. For a long this is
    below entry; for a short, above entry.
    """
    if leverage <= 0:
        raise ValueError("leverage must be > 0")
    s = _side_sign(side)
    # P/entry = 1 - s*(1/L) + s*0 ...  unified:
    #   long : entry*(1 - 1/L + mmr + fee_rate)
    #   short: entry*(1 + 1/L - mmr - fee_rate)
    return entry * (1.0 - s * (1.0 / leverage) + s * (mmr + fee_rate))


def liquidation_hit(entry: float, side: str, leverage: float,
                    bar_low: float, bar_high: float,
                    mmr: float = 0.005, fee_rate: float = 0.0) -> bool:
    """
    Did price reach the liquidation level within a bar [bar_low, bar_high]?
    Long  -> liquidated if bar_low  <= liq_price.
    Short -> liquidated if bar_high >= liq_price.
    """
    liq = liquidation_price(entry, side, leverage, mmr, fee_rate)
    if _side_sign(side) == 1:
        return bar_low <= liq
    return bar_high >= liq


def funding_pnl(side: str, notional: float, rates: Sequence[float]) -> float:
    """
    Signed funding PnL over the funding settlements the position was held
    across. `rates` = list of funding rates that settled during the hold.

    Positive rate => longs pay shorts, so:
        long  PnL = -notional * sum(rates)
        short PnL = +notional * sum(rates)
    Returns currency PnL (negative = paid out).
    """
    s = _side_sign(side)
    total_rate = float(sum(rates)) if rates else 0.0
    return -s * notional * total_rate


def net_perp_pnl(side: str, entry: float, exit_price: float, qty: float,
                 funding_rates: Sequence[float] = (),
                 fee_rate: float = 0.00055) -> float:
    """
    Convenience: full realized PnL of a perp trade in quote currency, including
    round-trip taker fees and funding accrued over `funding_rates`.

        price_pnl = sign * qty * (exit - entry)
        fees      = (entry + exit) * qty * fee_rate          (round trip)
        funding   = funding_pnl(side, entry*qty, funding_rates)
    """
    s = _side_sign(side)
    notional = entry * qty
    price_pnl = s * qty * (exit_price - entry)
    fees = (entry + exit_price) * qty * fee_rate
    fund = funding_pnl(side, notional, funding_rates)
    return price_pnl - fees + fund


if __name__ == '__main__':
    # quick manual sanity check
    print('liq long  10x:', liquidation_price(100, 'LONG', 10, 0.005))   # 90.5
    print('liq short 10x:', liquidation_price(100, 'SHORT', 10, 0.005))  # 109.5
    print('funding long :', funding_pnl('LONG', 1000, [0.0001, 0.0001, -0.0002]))  # 0.0
    print('notional     :', position_notional(10, 100, 98))              # 500.0
