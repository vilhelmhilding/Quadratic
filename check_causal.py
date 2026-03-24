#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PRICES_PATH = ROOT / "prices.csv"
ORDERS_PATH = ROOT / "orders.csv"


def run_algorithm() -> pd.DataFrame:
    cmd = (
        "import runpy, trading_simulator; "
        "trading_simulator.TradingSimulator.plot_performance = lambda *a, **k: None; "
        "runpy.run_path('algorithm.py')"
    )
    result = subprocess.run(
        [sys.executable, "-c", cmd],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"algorithm.py failed:\n{result.stdout}\n{result.stderr}")

    return pd.read_csv(ORDERS_PATH, parse_dates=["Date"])


def main() -> int:
    prices = pd.read_csv(PRICES_PATH)
    original_prices = prices.copy(deep=True)

    half_idx = len(prices) // 2
    cutoff_date = pd.to_datetime(prices.loc[half_idx, "Date"])

    change_row = min(half_idx + 10, len(prices) - 1)
    change_col = [c for c in prices.columns if c != "Date"][0]

    try:
        print("Run 1/2: baseline")
        base_orders = run_algorithm()

        prices.loc[change_row, change_col] = float(prices.loc[change_row, change_col]) * 1.1
        prices.to_csv(PRICES_PATH, index=False)

        print("Run 2/2: after changing future price")
        changed_orders = run_algorithm()
    finally:
        original_prices.to_csv(PRICES_PATH, index=False)

    base_prefix = base_orders[base_orders["Date"] <= cutoff_date].reset_index(drop=True)
    changed_prefix = changed_orders[changed_orders["Date"] <= cutoff_date].reset_index(drop=True)

    if base_prefix.equals(changed_prefix):
        print("PASS: Orders are identical up to halfway point.")
        return 0

    print("FAIL: Orders changed before halfway point (forward-looking behavior).")
    if len(base_prefix) != len(changed_prefix):
        print(f"Different number of orders before cutoff: {len(base_prefix)} vs {len(changed_prefix)}")
        return 1

    mismatches = base_prefix.ne(changed_prefix).any(axis=1)
    first = mismatches.idxmax()
    print("First mismatch:")
    print("baseline:", base_prefix.loc[first].to_dict())
    print("changed: ", changed_prefix.loc[first].to_dict())
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
