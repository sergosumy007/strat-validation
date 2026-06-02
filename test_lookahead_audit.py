"""
test_lookahead_audit.py — self-tests (SDET) for lookahead_audit.py.

Run:  python -m pytest test_lookahead_audit.py -q
"""

from lookahead_audit import audit_lookahead


def _write(tmp_path, body: str):
    p = tmp_path / "strat.py"
    p.write_text(body, encoding="utf-8")
    return str(p)


def _sev(result, code):
    for f in result["findings"]:
        if f["code"] == code:
            return f["severity"]
    return None


# A clean strategy: costs modelled, only past indexing, no lagging indicators.
_CLEAN = (
    "def detect_signals(df):\n"
    "    fee = 0.0006  # slippage and commission via cost_model\n"
    "    out = []\n"
    "    vals = df.values\n"
    "    for i, row in enumerate(vals):\n"
    "        if i > 0 and row[3] > vals[i-1][3]:\n"
    "            out.append(i)\n"
    "    return out\n"
)


def test_clean_file_passes(tmp_path):
    res = audit_lookahead(_write(tmp_path, _CLEAN))
    assert res["verdict"] == "PASS"
    assert res["n_fail"] == 0
    assert _sev(res, "LA-01") == "PASS"
    assert _sev(res, "LA-07") == "PASS"   # cost keywords present


def test_forward_index_fails(tmp_path):
    body = ("def f(arr):\n"
            "    z = 0\n"
            "    for i, x in enumerate(arr):\n"
            "        z = arr[i + 1]\n"   # future bar
            "    return z\n"
            "fee = 1  # slippage\n")
    res = audit_lookahead(_write(tmp_path, body))
    assert _sev(res, "LA-01") == "FAIL"
    assert res["verdict"] == "FAIL"


def test_negative_shift_fails(tmp_path):
    body = "import pandas as pd\nx = df['c'].shift(-1)  # fee slippage\n"
    res = audit_lookahead(_write(tmp_path, body))
    assert _sev(res, "LA-02") == "FAIL"
    assert res["verdict"] == "FAIL"


def test_at_future_index_fails(tmp_path):
    body = "y = df.at[i + 3, 'close']  # commission slippage\n"
    res = audit_lookahead(_write(tmp_path, body))
    assert _sev(res, "LA-03") == "FAIL"


def test_swing_window_fails(tmp_path):
    body = "w = arr[i-sw:i+sw]  # fee\n"
    res = audit_lookahead(_write(tmp_path, body))
    assert _sev(res, "LA-04") == "FAIL"


def test_missing_cost_model_fails(tmp_path):
    body = "def detect(df):\n    return [1, 2, 3]\n"   # no fee/slippage keyword
    res = audit_lookahead(_write(tmp_path, body))
    assert _sev(res, "LA-07") == "FAIL"
    assert res["verdict"] == "FAIL"


def test_rolling_without_shift_warns(tmp_path):
    body = "ma = df['close'].rolling(20).mean()  # fee slippage\n"
    res = audit_lookahead(_write(tmp_path, body))
    assert _sev(res, "LA-05") == "WARN"


def test_lagging_indicator_warns(tmp_path):
    body = "rsi = compute_rsi(df)  # fee slippage applied\n"
    res = audit_lookahead(_write(tmp_path, body))
    assert _sev(res, "LA-08") == "WARN"
    # WARN-only file (costs present, no leaks) -> overall WARN
    assert res["verdict"] == "WARN"


def test_syntax_error_fails(tmp_path):
    body = "def broken(:\n    pass\n"
    res = audit_lookahead(_write(tmp_path, body))
    assert res["verdict"] == "FAIL"
    assert _sev(res, "LA-00") == "FAIL"


def test_missing_file_errors():
    res = audit_lookahead("D:/MyScreener/__no_such_strategy__.py")
    assert res["verdict"] == "ERROR"
