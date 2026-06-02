"""
lookahead_audit.py — static code-integrity audit for look-ahead bias.

A backtest can show a beautiful Sharpe and still be worthless if the code peeks
into the future. This module performs a **static (AST + regex) scan** of a
strategy's source file and reports look-ahead / data-integrity findings.

It is engine-agnostic: point it at any Python strategy file. Checks are ported
from the project's Tier-1 audit (strategy_validator.py) but kept standalone so
the validation report has no heavy dependency.

Severities:  FAIL (hard leak)  ·  WARN (verify manually)  ·  INFO  ·  PASS

Public API:
    audit_lookahead(source_path) -> dict
        {'verdict', 'n_fail', 'n_warn', 'n_pass', 'findings': [...]}
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import List, Optional, Tuple

__all__ = ["audit_lookahead", "Finding", "grep"]


class Finding:
    """One audit result line."""
    __slots__ = ("severity", "code", "title", "detail", "hits")

    def __init__(self, severity: str, code: str, title: str,
                 detail: str, hits: Optional[List[Tuple[int, str]]] = None):
        self.severity = severity
        self.code = code
        self.title = title
        self.detail = detail
        self.hits = hits or []

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "code": self.code,
            "title": self.title,
            "detail": self.detail,
            "hits": [{"line": ln, "src": src.strip()} for ln, src in self.hits[:5]],
        }


def grep(lines: List[str], pattern: str, flags: int = 0) -> List[Tuple[int, str]]:
    """Return (1-based line number, line) for every regex match."""
    rx = re.compile(pattern, flags)
    return [(i + 1, l.rstrip("\n")) for i, l in enumerate(lines) if rx.search(l)]


class _ForwardIndexVisitor(ast.NodeVisitor):
    """Find arr[loop_var + k] (k > 0) inside enumerate() loops — future access."""

    def __init__(self, src_lines: List[str]):
        self.lines = src_lines
        self.hits: List[Tuple[int, str]] = []
        self._vars: List[set] = []

    def visit_For(self, node):
        enum_vars = set()
        if (isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "enumerate"):
            t = node.target
            if isinstance(t, ast.Tuple) and len(t.elts) >= 2:
                v = t.elts[0]
                if isinstance(v, ast.Name):
                    enum_vars.add(v.id)
            elif isinstance(t, ast.Name):
                enum_vars.add(t.id)
        self._vars.append(enum_vars)
        self.generic_visit(node)
        self._vars.pop()

    def visit_Subscript(self, node):
        all_vars = set().union(*self._vars) if self._vars else set()
        if not all_vars:
            self.generic_visit(node)
            return
        sl = node.slice
        if isinstance(sl, ast.BinOp) and isinstance(sl.op, ast.Add):
            if (isinstance(sl.left, ast.Name) and sl.left.id in all_vars
                    and isinstance(sl.right, ast.Constant)
                    and isinstance(sl.right.value, (int, float))
                    and sl.right.value > 0):
                ln = node.lineno
                if 0 < ln <= len(self.lines):
                    self.hits.append((ln, self.lines[ln - 1]))
        self.generic_visit(node)


def audit_lookahead(source_path) -> dict:
    """
    Statically audit a strategy source file for look-ahead bias and integrity.

    Returns a dict with an overall ``verdict`` (FAIL > WARN > PASS), counts and
    a list of findings. A FAIL means the backtest cannot be trusted as-is.
    """
    path = Path(source_path)
    if not path.exists():
        return {"verdict": "ERROR", "n_fail": 0, "n_warn": 0, "n_pass": 0,
                "findings": [Finding("FAIL", "LA-00", "Source not found",
                                     f"File does not exist: {path}").to_dict()]}

    src = path.read_text(encoding="utf-8", errors="replace")
    lines = src.splitlines()
    findings: List[Finding] = []

    # LA-00 — parses at all
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return {"verdict": "FAIL", "n_fail": 1, "n_warn": 0, "n_pass": 0,
                "findings": [Finding("FAIL", "LA-00", "Python syntax error",
                                     f"ast.parse failed: {e}").to_dict()]}

    # LA-01 — forward indexing arr[i+k] in enumerate loops
    vis = _ForwardIndexVisitor(lines)
    vis.visit(tree)
    if vis.hits:
        findings.append(Finding("FAIL", "LA-01", "Forward indexing look-ahead",
            "arr[i+k] inside an enumerate loop reads future bars.", vis.hits))
    else:
        findings.append(Finding("PASS", "LA-01", "Forward indexing look-ahead",
            "No arr[i+k] future access in enumerate loops."))

    # LA-02 — negative shift(): pulls future into the present
    shift_hits = grep(lines, r"\.shift\(\s*-\d", re.IGNORECASE)
    if shift_hits:
        findings.append(Finding("FAIL", "LA-02", "Negative shift(-N)",
            "shift(-N) moves future values backward in time.", shift_hits))
    else:
        findings.append(Finding("PASS", "LA-02", "Negative shift(-N)",
            "No negative .shift() found."))

    # LA-03 — .at[i+N] / .iat[i+N] future index access
    at_hits = grep(lines, r"\.(at|iat)\s*\[\s*\w+\s*\+\s*\d+", re.IGNORECASE)
    if at_hits:
        findings.append(Finding("FAIL", "LA-03", ".at[i+N] future access",
            "Direct .at/.iat access to a future bar index.", at_hits))
    else:
        findings.append(Finding("PASS", "LA-03", ".at[i+N] future access",
            "No .at[i+N] / .iat[i+N] future indexing."))

    # LA-04 — swing window [i-sw : i+sw] includes future bars
    sw_hits = grep(lines, r"\[\s*i\s*-\s*sw\s*:\s*i\s*\+\s*sw", re.IGNORECASE)
    sw_hits = [(ln, l) for ln, l in sw_hits
               if not l.strip().startswith("#") and '"""' not in l and "'''" not in l]
    if sw_hits:
        findings.append(Finding("FAIL", "LA-04", "Swing window [i-sw:i+sw]",
            "Window [i-sw:i+sw+1] includes future bars; should be [i-sw:i+1].", sw_hits))
    else:
        findings.append(Finding("PASS", "LA-04", "Swing window look-ahead",
            "No [i-sw:i+sw] symmetric windows found."))

    # LA-05 — rolling() without .shift(1): may include the current bar (WARN)
    roll = grep(lines, r"\.rolling\([^)]+\)\.(mean|std|sum|max|min)\(\)", re.IGNORECASE)
    roll = [(ln, l) for ln, l in roll if ".shift(" not in l and "rolling_shift" not in l]
    if roll:
        findings.append(Finding("WARN", "LA-05", "rolling() without .shift(1)",
            "Rolling aggregates without a shift may include the current (open) bar.",
            roll[:10]))
    else:
        findings.append(Finding("PASS", "LA-05", "rolling() with shift",
            "All rolling() aggregates use .shift()."))

    # LA-06 — access to close[-1] / iloc[-1] (verify the bar is closed) (WARN)
    close_hits = grep(lines,
        r"close\s*\[\s*['\"]?-1['\"]?\s*\]|\.iloc\s*\[\s*-1\s*\]\s*\[.*close",
        re.IGNORECASE)
    if close_hits:
        findings.append(Finding("WARN", "LA-06", "Access to close[-1] / iloc[-1]",
            "Verify the last bar is actually closed when its close is used.",
            close_hits[:5]))
    else:
        findings.append(Finding("PASS", "LA-06", "Current-bar close access",
            "No suspicious unclosed-bar close access."))

    # LA-07 — transaction costs present (missing costs => inflated backtest) (FAIL)
    cost_hits = grep(lines,
        r"calc_fixed_rr_outcomes|cost_model|apply_costs|FEE_RT|slippage|commission|fee",
        re.IGNORECASE)
    if cost_hits:
        findings.append(Finding("PASS", "LA-07", "Transaction costs modelled",
            "References to cost model found (fees / slippage accounted for).",
            cost_hits[:3]))
    else:
        findings.append(Finding("FAIL", "LA-07", "No transaction-cost model",
            "No fees/slippage found — the backtest is optimistic / overstated."))

    # LA-08 — lagging indicators that don't predict the future (WARN)
    lag = [(r"\brsi\b", "RSI"), (r"\bmacd\b", "MACD"), (r"\bstoch\w*", "Stochastic"),
           (r"\bwilliams\b", "Williams %R"), (r"\bcci\b", "CCI"), (r"\bmfi\b", "MFI")]
    lag_found = []
    for pat, name in lag:
        hits = [(ln, l) for ln, l in grep(lines, pat, re.IGNORECASE)
                if not re.match(r"\s*#", l)]
        if hits:
            lag_found.append((name, hits))
    if lag_found:
        names = ", ".join(n for n, _ in lag_found)
        findings.append(Finding("WARN", "LA-08", f"Lagging indicators: {names}",
            "RSI/MACD/Stochastic are lagging and do not predict future returns.",
            lag_found[0][1][:3]))
    else:
        findings.append(Finding("PASS", "LA-08", "No lagging indicators",
            "No RSI/MACD/Stochastic/Williams/CCI/MFI found."))

    n_fail = sum(1 for f in findings if f.severity == "FAIL")
    n_warn = sum(1 for f in findings if f.severity == "WARN")
    n_pass = sum(1 for f in findings if f.severity == "PASS")
    verdict = "FAIL" if n_fail else ("WARN" if n_warn else "PASS")

    return {
        "verdict": verdict,
        "n_fail": n_fail,
        "n_warn": n_warn,
        "n_pass": n_pass,
        "findings": [f.to_dict() for f in findings],
    }
