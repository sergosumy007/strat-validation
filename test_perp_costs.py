"""Unit tests for perp_costs.py — funding / margin / liquidation modeling."""
import math
import pytest

from perp_costs import (position_notional, maintenance_margin, liquidation_price,
                        liquidation_hit, funding_pnl, net_perp_pnl)


def approx(a, b, tol=1e-9):
    return math.isclose(a, b, rel_tol=0, abs_tol=tol)


# ── liquidation_price ───────────────────────────────────────────────────────
def test_liq_long_basic():
    # entry 100, 10x, mmr 0.5%, no fee -> 100*(1-0.1+0.005)=90.5
    assert approx(liquidation_price(100, 'LONG', 10, 0.005), 90.5)


def test_liq_short_basic():
    assert approx(liquidation_price(100, 'SHORT', 10, 0.005), 109.5)


def test_liq_long_below_short_above_entry():
    assert liquidation_price(100, 'LONG', 5, 0.005) < 100
    assert liquidation_price(100, 'SHORT', 5, 0.005) > 100


def test_liq_higher_leverage_closer_to_entry():
    # 20x liq is closer to entry than 5x liq (long)
    near = liquidation_price(100, 'LONG', 20, 0.005)
    far = liquidation_price(100, 'LONG', 5, 0.005)
    assert near > far  # closer to 100 from below


def test_liq_fee_widens_toward_entry():
    no_fee = liquidation_price(100, 'LONG', 10, 0.005, fee_rate=0.0)
    with_fee = liquidation_price(100, 'LONG', 10, 0.005, fee_rate=0.0006)
    assert with_fee > no_fee  # long liq moves up (gets hit sooner)


def test_liq_invalid_leverage():
    with pytest.raises(ValueError):
        liquidation_price(100, 'LONG', 0, 0.005)


def test_liq_invalid_side():
    with pytest.raises(ValueError):
        liquidation_price(100, 'flat', 10, 0.005)


# ── liquidation_hit ─────────────────────────────────────────────────────────
def test_liq_hit_long():
    # liq = 90.5; a bar dipping to 90 hits it, a bar low 91 does not
    assert liquidation_hit(100, 'LONG', 10, bar_low=90.0, bar_high=101)
    assert not liquidation_hit(100, 'LONG', 10, bar_low=91.0, bar_high=101)


def test_liq_hit_short():
    # liq = 109.5; bar high 110 hits, 109 does not
    assert liquidation_hit(100, 'SHORT', 10, bar_low=99, bar_high=110.0)
    assert not liquidation_hit(100, 'SHORT', 10, bar_low=99, bar_high=109.0)


def test_low_leverage_not_liquidated_at_sl_distance():
    # 2x long, liq ~ 50.5; a normal 2% SL never triggers liquidation
    assert not liquidation_hit(100, 'LONG', 2, bar_low=98.0, bar_high=101)


# ── funding_pnl ─────────────────────────────────────────────────────────────
def test_funding_zero_when_rates_cancel():
    assert approx(funding_pnl('LONG', 1000, [0.0001, 0.0001, -0.0002]), 0.0)


def test_funding_long_pays_positive_rate():
    # positive funding => long pays => negative PnL
    assert approx(funding_pnl('LONG', 1000, [0.001]), -1.0)


def test_funding_short_receives_positive_rate():
    assert approx(funding_pnl('SHORT', 1000, [0.001]), 1.0)


def test_funding_empty_rates():
    assert approx(funding_pnl('LONG', 1000, []), 0.0)


def test_funding_sign_symmetry():
    rates = [0.0003, -0.0001, 0.0002]
    assert approx(funding_pnl('LONG', 500, rates),
                  -funding_pnl('SHORT', 500, rates))


# ── maintenance_margin / position_notional ─────────────────────────────────
def test_maintenance_margin():
    assert approx(maintenance_margin(1000, 0.005), 5.0)


def test_maintenance_margin_tier_deduction_floored():
    # huge deduction cannot make MM negative
    assert maintenance_margin(1000, 0.005, maint_amount=99) == 0.0


def test_position_notional():
    # risk 10, entry 100, sl 98 -> qty 5, notional 500
    assert approx(position_notional(10, 100, 98), 500.0)


def test_position_notional_degenerate():
    assert position_notional(10, 100, 100) == 0.0  # zero SL distance


# ── net_perp_pnl ────────────────────────────────────────────────────────────
def test_net_perp_pnl_long_no_costs():
    # qty 1, 100->110, no fee, no funding -> +10
    assert approx(net_perp_pnl('LONG', 100, 110, 1, fee_rate=0.0), 10.0)


def test_net_perp_pnl_includes_fees_and_funding():
    # qty 1, 100->110, fee 0.001 round trip on (100+110)=210 -> 0.21 fee
    # funding [0.001] long pays -> -0.1 on notional 100
    pnl = net_perp_pnl('LONG', 100, 110, 1, funding_rates=[0.001], fee_rate=0.001)
    assert approx(pnl, 10.0 - 0.21 - 0.1)


def test_net_perp_pnl_short_profit_on_drop():
    assert approx(net_perp_pnl('SHORT', 100, 90, 1, fee_rate=0.0), 10.0)


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
