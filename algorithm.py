import numpy as np
import pandas as pd
from pathlib import Path
from trading_simulator import TradingSimulator

# ===== LOAD DATA =====
prices = pd.read_csv("prices.csv", index_col="Date", parse_dates=True)

STOCK_COLS = [c for c in prices.columns if c.startswith("Stock_")]
FX_COLS    = [c for c in prices.columns if c.startswith("FX_")]

print(f"Loaded {len(prices)} trading days, {len(prices.columns)} assets")

# ===== ASSET METADATA =====
equity_ccy = {
    'Stock_01': 'Crncy_03', 'Stock_02': 'Crncy_04', 'Stock_03': 'Crncy_04',
    'Stock_04': 'Crncy_02', 'Stock_05': 'Crncy_03', 'Stock_06': 'Crncy_02',
    'Stock_07': 'Crncy_03', 'Stock_08': 'Crncy_02', 'Stock_09': 'Crncy_04',
    'Stock_10': 'Crncy_03', 'Stock_11': 'Crncy_01', 'Stock_12': 'Crncy_04',
    'Stock_13': 'Crncy_01', 'Stock_14': 'Crncy_01', 'Stock_15': 'Crncy_01',
}

fx_pairs_map = {
    'FX_01': ('Crncy_02', 'Crncy_01'),
    'FX_02': ('Crncy_04', 'Crncy_02'),
    'FX_03': ('Crncy_04', 'Crncy_03'),
    'FX_04': ('Crncy_02', 'Crncy_03'),
    'FX_05': ('Crncy_01', 'Crncy_03'),
    'FX_06': ('Crncy_04', 'Crncy_01'),
}

# ===== STRATEGY =====
TREND_WINDOW   = 50
VOL_WINDOW     = 100
RISK_WINDOW    = 100

def compute_positions(stock_prices):
    daily_returns = stock_prices.pct_change()

    signal     = np.sign(daily_returns.rolling(TREND_WINDOW).sum())
    volatility = np.sqrt(
        (daily_returns ** 2).rolling(VOL_WINDOW, min_periods=VOL_WINDOW // 2).sum()
    )
    vol_scaled_pos = signal / volatility.replace(0, np.nan)

    portfolio_daily_pnl = (vol_scaled_pos.shift(2) * daily_returns).dropna(how='all').sum(axis=1)
    portfolio_risk      = portfolio_daily_pnl.rolling(RISK_WINDOW, min_periods=20).std()

    return vol_scaled_pos.div(portfolio_risk, axis=0) 

target_shares_df = compute_positions(prices[STOCK_COLS])

def strategy(row_pos, cash, portfolio, signal_prices, data):
    orders = []
    date   = data.index[row_pos]

    if date not in target_shares_df.index:
        return orders

    targets = target_shares_df.loc[date]
    if targets.isna().all():
        return orders

    total_equity_notional = 0.0
    for ticker in STOCK_COLS:
        tgt = targets.get(ticker)
        if pd.isna(tgt):
            continue
        tgt   = int(round(tgt))
        delta = tgt - portfolio.get(ticker, 0)
        if delta != 0:
            orders.append(('BUY' if delta > 0 else 'SELL', ticker, abs(delta)))
        total_equity_notional += abs(tgt * signal_prices[ticker])

    hedge_per_pair = total_equity_notional / len(FX_COLS)
    for fx_ticker in FX_COLS:
        fx_price = signal_prices.get(fx_ticker, 100.0)
        tgt_fx   = -int(round(hedge_per_pair / fx_price))
        delta_fx = tgt_fx - portfolio.get(fx_ticker, 0)
        if delta_fx != 0:
            orders.append(('BUY' if delta_fx > 0 else 'SELL', fx_ticker, abs(delta_fx)))

    return orders

# ===== RUN SIMULATION =====
simulator = TradingSimulator(
    assets              = list(prices.columns),
    initial_cash        = 100_000,
    equity_currency_map = equity_ccy,
    fx_pairs_map        = fx_pairs_map,
)
simulator.run(strategy, prices, prices)
simulator.save_results(
    orders_file    =  "orders.csv",
    portfolio_file =  "portfolio.csv",
)
simulator.plot_performance(prices, save_file="performance_plot.png")
