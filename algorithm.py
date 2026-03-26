import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trading_simulator import TradingSimulator


prices = pd.read_csv("prices_real_dates.csv", index_col="Date", parse_dates=["Date"])
ret = prices.diff()
log_ret = np.log(prices).diff()

equity_ccy = {
    "Stock_01": "Crncy_03",
    "Stock_02": "Crncy_04",
    "Stock_03": "Crncy_04",
    "Stock_04": "Crncy_02",
    "Stock_05": "Crncy_03",
    "Stock_06": "Crncy_02",
    "Stock_07": "Crncy_03",
    "Stock_08": "Crncy_02",
    "Stock_09": "Crncy_04",
    "Stock_10": "Crncy_03",
    "Stock_11": "Crncy_01",
    "Stock_12": "Crncy_04",
    "Stock_13": "Crncy_01",
    "Stock_14": "Crncy_01",
    "Stock_15": "Crncy_01",
}

fx_pairs_map = {
    "FX_01": ("Crncy_02", "Crncy_01"),
    "FX_02": ("Crncy_04", "Crncy_02"),
    "FX_03": ("Crncy_04", "Crncy_03"),
    "FX_04": ("Crncy_02", "Crncy_03"),
    "FX_05": ("Crncy_01", "Crncy_03"),
    "FX_06": ("Crncy_04", "Crncy_01"),
}

initial_cash = 100_000
leverage = 5
window = 30

# Capital split
sleeve_1_cash = initial_cash * 0.6
sleeve_2_cash0 = initial_cash * 0.2
sleeve_3_cash0 = initial_cash * 0.2


# Strategy 1
def build_pair(p1, p2, log_ret, window):
    x = log_ret[p2]
    y = log_ret[p1]
    beta = (y.rolling(window).cov(x) / x.rolling(window).var()).replace(
        [np.inf, -np.inf], np.nan
    )

    residual = y - beta * x
    resid_std = residual.rolling(window).std()
    cap = 1 * resid_std
    residual_capped = residual.clip(lower=-cap, upper=cap)
    spread = residual_capped.cumsum()
    z = (
        (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    ).fillna(0)

    return p1, p2, beta, z


px = prices.reindex(log_ret.index)

pairs = [
    build_pair("Idx_02", "Idx_03", log_ret, window),
    build_pair("Idx_01", "Idx_03", log_ret, window),
    build_pair("Idx_01", "Idx_04", log_ret, window),
    build_pair("Idx_01", "Stock_01", log_ret, window),
    build_pair("Idx_01", "Stock_06", log_ret, window),
    build_pair("Idx_01", "Stock_08", log_ret, window),
    build_pair("Idx_01", "Stock_10", log_ret, window),
    build_pair("Idx_01", "Stock_11", log_ret, window),
    build_pair("Idx_01", "Stock_14", log_ret, window),
]

states_1 = {i: 0 for i in range(len(pairs))}


def target_strategy_1(row_pos, data):
    date = data.index[row_pos]
    if date not in px.index:
        return {}

    target_shares = {}
    n_strats = len(pairs)
    capital_per_strat = (sleeve_1_cash * leverage) / n_strats

    for i, (p1, p2, beta, z) in enumerate(pairs):
        if pd.isna(px.loc[date, p1]) or pd.isna(px.loc[date, p2]):
            continue

        z_t = z.loc[date]
        beta_now = beta.loc[date] if pd.notna(beta.loc[date]) else 1.0
        state = states_1[i]

        if state == 0:
            if p1 == "Idx_01":
                if z_t > 2.5:
                    state = -1
                elif z_t < -1.0:
                    state = 1
            else:
                if z_t > 1.0:
                    state = -1
                elif z_t < -1.0:
                    state = 1
        elif abs(z_t) < 0.5:
            state = 0

        states_1[i] = state

        if state == 0:
            tgt1 = 0.0
        else:
            tgt1 = state * (capital_per_strat / px.loc[date, p1])

        tgt2 = -tgt1 * beta_now * px.loc[date, p1] / px.loc[date, p2]

        target_shares[p1] = target_shares.get(p1, 0.0) + tgt1
        target_shares[p2] = target_shares.get(p2, 0.0) + tgt2

    return target_shares


# Strategy 2
STOCK_COLS = [c for c in prices.columns if c.startswith("Stock_")]
FX_COLS = [c for c in prices.columns if c.startswith("FX_")]

LOOKBACK = 30
TOP_N = 4

signal = prices[STOCK_COLS].pct_change(LOOKBACK).resample("ME").last()
rebalance_dates = set(signal.index)
currencies = sorted(set(equity_ccy.values()))
fx_list = list(FX_COLS)

A = np.zeros((len(currencies), len(fx_list)))
for j, fx_ticker in enumerate(fx_list):
    c1, c2 = fx_pairs_map[fx_ticker]
    A[currencies.index(c1), j] = 1.0
    A[currencies.index(c2), j] = -1.0

sleeve_2_cash = sleeve_2_cash0
sleeve_2_positions = {}


def mark_to_market(positions, signal_prices):
    return sum(q * signal_prices.get(t, 0) for t, q in positions.items())


def target_strategy_2(row_pos, cash, portfolio, signal_prices, data):
    global sleeve_2_cash, sleeve_2_positions

    date = data.index[row_pos]

    if date not in rebalance_dates:
        return dict(sleeve_2_positions)

    top = signal.loc[date].nlargest(TOP_N).index

    total_value = sleeve_2_cash + sum(
        sleeve_2_positions.get(t, 0) * signal_prices.get(t, 0) for t in prices.columns
    )
    target_per_stock = total_value / TOP_N

    targets = {}
    ccy_exposure = {ccy: 0.0 for ccy in currencies}

    for ticker in STOCK_COLS:
        px = signal_prices.get(ticker, 0)
        tgt = int(target_per_stock / px) if ticker in top and px > 0 else 0
        targets[ticker] = tgt

        if tgt != 0 and px > 0:
            ccy_exposure[equity_ccy[ticker]] += tgt * px

    idx_ticker = "Idx_04"
    px_idx = signal_prices.get(idx_ticker, 0)

    if px_idx > 0:
        total_beta_notional = 0.0
        idx_ret = prices[idx_ticker].pct_change()

        for ticker in STOCK_COLS:
            q = targets.get(ticker, 0)
            px = signal_prices.get(ticker, 0)
            if q == 0 or px <= 0:
                continue

            stk_ret = prices[ticker].pct_change()
            beta = (
                stk_ret.rolling(LOOKBACK).cov(idx_ret) / idx_ret.rolling(LOOKBACK).var()
            ).loc[date]

            beta = beta if pd.notna(beta) else 1.0
            total_beta_notional += beta * q * px

        idx_shares = int(-total_beta_notional / px_idx)
        targets[idx_ticker] = idx_shares

    trade_cash = 0.0
    all_tickers = set(sleeve_2_positions) | set(targets)

    for t in all_tickers:
        old_q = sleeve_2_positions.get(t, 0)
        new_q = targets.get(t, 0)
        delta = new_q - old_q
        px = signal_prices.get(t, 0)
        trade_cash -= delta * px

    sleeve_2_cash += trade_cash
    sleeve_2_positions = dict(targets)

    return dict(sleeve_2_positions)


# Strategy 3
STOCKS_3 = [f"Stock_{i:02d}" for i in range(1, 16)]
ETF_3 = "Idx_04"

LOOKBACK_3 = 300
ENTRY_Z_3 = 2.0
EXIT_Z_3 = 0.0
MAX_Z_3 = 4.0
MAX_GROSS_3 = leverage
MIN_STD_3 = 1e-8

init_prices_3 = prices[STOCKS_3].iloc[0]
init_shares_3 = (1.0 / len(STOCKS_3)) / init_prices_3

synthetic_3 = prices[STOCKS_3].mul(init_shares_3, axis=1).sum(axis=1)
synthetic_3 = synthetic_3 / synthetic_3.iloc[0] * prices[ETF_3].iloc[0]

mispricing_3 = np.log(prices[ETF_3] / synthetic_3)
mu_3 = mispricing_3.rolling(LOOKBACK_3).mean()
sigma_3 = mispricing_3.rolling(LOOKBACK_3).std().clip(lower=MIN_STD_3)
zscore_3 = ((mispricing_3 - mu_3) / sigma_3).replace([np.inf, -np.inf], np.nan)

rebalance_dates_3 = set(zscore_3.dropna().index)

sleeve_3_cash = sleeve_3_cash0
sleeve_3_positions = {}


def _equity_3(cash, positions, signal_prices):
    return cash + sum(
        positions.get(t, 0) * signal_prices.get(t, 0) for t in prices.columns
    )


def _current_targets_3(positions):
    return {
        t: positions.get(t, 0) for t in [ETF_3] + STOCKS_3 if positions.get(t, 0) != 0
    }


def _cap_targets_to_cash_3(targets, cash, portfolio, signal_prices):
    buy_cost = 0.0
    buy_deltas = {}

    for t in [ETF_3] + STOCKS_3:
        curr = portfolio.get(t, 0)
        tgt = targets.get(t, 0)
        delta = tgt - curr
        px = signal_prices.get(t, np.nan)

        if delta > 0 and np.isfinite(px) and px > 0:
            cost = delta * px
            buy_deltas[t] = (delta, px)
            buy_cost += cost

    if buy_cost <= cash or buy_cost <= 0:
        return targets

    scale = cash / buy_cost
    capped = targets.copy()

    for t, (delta, px) in buy_deltas.items():
        curr = portfolio.get(t, 0)
        capped_delta = int(delta * scale)
        capped[t] = curr + capped_delta

    return capped


def target_strategy_3(row_pos, signal_prices, data):
    global sleeve_3_cash, sleeve_3_positions

    date = data.index[row_pos]
    curr = _current_targets_3(sleeve_3_positions)

    if date not in rebalance_dates_3 or pd.isna(zscore_3.loc[date]):
        return curr

    z = float(zscore_3.loc[date])

    px_etf = signal_prices.get(ETF_3, np.nan)
    px_stk = pd.Series({t: signal_prices.get(t, np.nan) for t in STOCKS_3}, dtype=float)

    if (
        not np.isfinite(px_etf)
        or px_etf <= 0
        or px_stk.isna().any()
        or (px_stk <= 0).any()
    ):
        targets = {t: 0 for t in [ETF_3] + STOCKS_3}
    elif abs(z) <= EXIT_Z_3:
        targets = {t: 0 for t in [ETF_3] + STOCKS_3}
    elif abs(z) < ENTRY_Z_3:
        targets = curr
    else:
        total_value = _equity_3(sleeve_3_cash, sleeve_3_positions, signal_prices)
        gross_target = total_value * MAX_GROSS_3 * min(abs(z), MAX_Z_3) / MAX_Z_3

        etf_notional = gross_target / 2.0
        basket_notional = gross_target / 2.0

        basket_value_now = (init_shares_3 * px_stk).sum()
        basket_weights_now = (init_shares_3 * px_stk) / basket_value_now

        etf_shares = int(etf_notional / px_etf)
        stock_shares = ((basket_notional * basket_weights_now) / px_stk).astype(int)

        targets = {}
        if z > 0:
            targets[ETF_3] = -etf_shares
            for t in STOCKS_3:
                targets[t] = int(stock_shares[t])
        else:
            targets[ETF_3] = etf_shares
            for t in STOCKS_3:
                targets[t] = -int(stock_shares[t])

        targets = _cap_targets_to_cash_3(
            targets, sleeve_3_cash, sleeve_3_positions, signal_prices
        )

    trade_cash = 0.0
    all_tickers = set(sleeve_3_positions) | set(targets)

    for t in all_tickers:
        old_q = sleeve_3_positions.get(t, 0)
        new_q = targets.get(t, 0)
        delta = new_q - old_q
        px = signal_prices.get(t, 0)
        trade_cash -= delta * px

    sleeve_3_cash += trade_cash
    sleeve_3_positions = dict(targets)

    return dict(sleeve_3_positions)


def apply_portfolio_fx_hedge(total_targets, signal_prices):
    targets = dict(total_targets)

    for fx in FX_COLS:
        targets[fx] = 0

    exp_02 = 0.0
    exp_03 = 0.0
    exp_04 = 0.0

    for ticker, q in targets.items():
        if ticker in fx_pairs_map:
            continue

        px = signal_prices.get(ticker, 0)
        if px <= 0:
            continue

        ccy = equity_ccy.get(ticker, "Crncy_01")
        val = q * px

        if ccy == "Crncy_02":
            exp_02 += val
        elif ccy == "Crncy_03":
            exp_03 += val
        elif ccy == "Crncy_04":
            exp_04 += val

    px_fx01 = signal_prices.get("FX_01", 0)
    px_fx05 = signal_prices.get("FX_05", 0)
    px_fx06 = signal_prices.get("FX_06", 0)

    if px_fx01 > 0:
        targets["FX_01"] = int(round(-exp_02 / px_fx01))

    if px_fx05 > 0:
        targets["FX_05"] = int(round(exp_03 / px_fx05))

    if px_fx06 > 0:
        targets["FX_06"] = int(round(-exp_04 / px_fx06))

    return targets


# Combined strategy
def strategy(row_pos, cash, portfolio, signal_prices, data):
    tgt1 = target_strategy_1(row_pos, data)
    tgt2 = target_strategy_2(row_pos, cash, portfolio, signal_prices, data)
    tgt3 = target_strategy_3(row_pos, signal_prices, data)

    total_targets = {}
    for d in (tgt1, tgt2, tgt3):
        for t, q in d.items():
            total_targets[t] = total_targets.get(t, 0.0) + q

    total_targets = apply_portfolio_fx_hedge(total_targets, signal_prices)

    ordered_tickers = list(prices.columns)
    sell_orders = []
    buy_orders = []

    for t in ordered_tickers:
        tgt = int(round(total_targets.get(t, 0.0)))
        current = portfolio.get(t, 0)
        delta = tgt - current

        if delta < 0:
            sell_orders.append(("SELL", t, abs(delta)))
        elif delta > 0:
            buy_orders.append(("BUY", t, abs(delta)))

    orders = sell_orders + buy_orders
    return orders


# Simulation
simulator = TradingSimulator(
    assets=list(prices.columns),
    initial_cash=initial_cash,
    equity_currency_map=equity_ccy,
    fx_pairs_map=fx_pairs_map,
)

simulator.run(strategy, prices, prices)
simulator.save_results("orders.csv", "portfolio.csv")
simulator.plot_performance(prices, save_file="performance_plot.png")
