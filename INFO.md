# Trading Algorithm Simulator

A Python-based trading simulation framework for testing and visualizing algorithmic trading strategies.

## Installation
Here's an installation guide. However honestly speaking, just ask ChatGPT/Claude/your-preferred-LLM to help you with installation by copy-pasting this text + specifying your computer's operating system.
The following python libraries will be needed: 

- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `jupyter` - Interactive notebooks (OPTIONAL)


### Step 1: If you are new to python and DON'T have `uv`, `pip`/`pip3` here's an installation guide for `uv` (Package Manager)
# If you have `uv` or `pip`, you can move on to Step 2. You can double check by running the commands below
```bash
pip3 -version
uv -version
```

`uv` is a fast Python package manager. Install it for your platform:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (using pip):**
```bash
pip install uv
```

### Step 2: Install Dependencies

Navigate to the project directory and install all required packages:

```bash
uv sync
```

This will install:
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `jupyter` - Interactive notebooks

**Note:** If you don't have a `pyproject.toml` or prefer manual installation:
```bash
uv pip install pandas numpy matplotlib jupyter
```

**Alternative using only pip:**
```bash
pip install pandas numpy matplotlib jupyter
pip3 install pandas numpy matplotlib jupyter

```

Let us know in case you have problems with installing the dependencies.

## Quick Start

### Option 1: Full Run (Recommended for Quick Results)

Run the complete simulation with visualization:

```bash
uv run algorithm.py      # Run strategy and generate plots
```

This will:
- Load all asset prices from `participant_data/prices.csv` (stocks, indices, commodities, currencies)
- Execute the trading strategy
- Generate `orders.csv` (all trades)
- Generate `portfolio.csv` (daily portfolio snapshots)
- Create `performance_plot.png` (comprehensive visualization)
- Display performance metrics (Sharpe ratio, profit/loss, etc.)

### Option 2: Interactive Exploration (Recommended for Learning)

First, make sure Jupyter is installed:
```bash
uv pip install jupyter
# OR with regular pip
pip install jupyter
```

Then walk through the simulation step-by-step:

**Option A: Open in browser**
```bash
uv run jupyter notebook Quickstart.ipynb
# OR if using pip
jupyter notebook Quickstart.ipynb
```

**Option B: Open directly in VS Code**
- Open `Quickstart.ipynb` in VS Code
- VS Code will prompt you to install the Jupyter extension if not already installed
- Click "Run All" or run cells individually with Shift+Enter

The notebook contains:
- A suggested Moving Average (MA) crossover algorithm
- Step-by-step data exploration
- Interactive signal generation
- Portfolio simulation with the `TradingSimulator` class

Run each cell to understand how the framework works before building your own strategy.

## Files Overview

- **`prices.csv`** - All asset prices in one file (columns: `Stock_01…`, `Idx_01…`, `Comm_01…`, `FX_01…`), indexed to 100 at the first training day
- **`algorithm.py`** - Complete trading strategy implementation (MA crossover example)
- **`Quickstart.ipynb`** - Interactive Jupyter notebook for learning and experimentation
- **`trading_simulator.py`** - Core simulation engine (DO NOT MODIFY - see below)
- **`check_causal.py`** - Forward-looking bias checker (see below)

## Checking for Forward-Looking Bias

Run `check_causal.py` to verify your strategy does not accidentally use future data:

```bash
uv run check_causal.py
# OR
python check_causal.py
```

The script runs your `algorithm.py` twice — once with the original prices and once with a small price change applied to a date in the second half of the dataset. It then checks that all orders up to the halfway point are **identical** in both runs. If they differ, your strategy has forward-looking bias.

- **PASS** — orders before the cutoff are unchanged; no forward-looking bias detected
- **FAIL** — orders before the cutoff changed; your strategy is likely peeking at future data

## Trade Execution

Trades are executed at **T+1 prices** — a signal generated on day T is filled at the price of day T+1, to account for the fact that end-of-day prices used to generate signals are not tradeable until the next day.

## Generated Outputs

- **`orders.csv`** - Trade execution history (Date, Ticker, Action, Shares, Price, Total)
- **`portfolio.csv`** - Daily portfolio snapshots (Date, Cash, Holdings, Total Value)
- **`performance_plot.png`** - Performance dashboard with three sections:
  - **Top**: Cumulative P&L (%) over the full period
  - **Middle**: Daily buy/sell trade counts
  - **Bottom-left**: Rolling 1-year window charts — Sharpe ratio, FX hedge score, and volatility (rolling std of daily % returns)
  - **Bottom-right**: Asset allocation pie (by total traded dollar volume, grouped by asset class) and Key Figures table

## Building Your Own Strategy

### Important: Do NOT Edit `trading_simulator.py`

The `TradingSimulator` class in `trading_simulator.py` is the **original reference implementation**.

**To customize the simulator:**

1. Create `custom_trading_simulator.py`
2. Define your `CustomTradingSimulator` class
3. Extend or modify the original `TradingSimulator`
4. Use your custom class in your algorithms

Example:
```python
from trading_simulator import TradingSimulator

class CustomTradingSimulator(TradingSimulator):
    def __init__(self, stocks, initial_cash=100000):
        super().__init__(stocks, initial_cash)
        # Add your custom initialization

    def execute_trade(self, date, ticker, signal, price):
        # Add your custom trading logic
        # e.g., dynamic position sizing, stop losses, etc.
        return super().execute_trade(date, ticker, signal, price)
```

## Performance Metrics

The simulator calculates and displays the following metrics:

| Metric | Description |
|---|---|
| **Net P&L** | Total gain/loss in portfolio currency units and as % of initial capital |
| **Sharpe** | Annualised Sharpe ratio |
| **Adj. Sharpe** | Sharpe penalised for skewness and fat tails |
| **Sharpe 1d lag** | Sharpe computed on returns shifted by one day — a large gap vs. Sharpe suggests lookahead bias |
| **Skewness** | Asymmetry of daily P&L; negative = occasional large losses with mostly small gains, positive = occasional large gains with mostly small losses |
| **Max DD** | Maximum peak-to-trough cumulative loss in portfolio currency |
| **FX Hedge Score** | Score (0–100) measuring how well FX exposure is hedged |
| **# Trades** | Total number of executed orders |


## Data Description

All assets are **anonymized** and all prices are **normalized to 100 on the first trading day** (e.g. a price of 150 means +50% since the start).

### Asset classes

| Asset class | Tickers | Currency |
|---|---|---|
| Equities | `Stock_01` – `Stock_15` | See table below |
| Indices | `Idx_01` – `Idx_04` | No associated currency |
| Commodities | `Comm_01` – `Comm_06` | No associated currency |
| FX pairs | `FX_01` – `FX_06` | — |

### Equity currency denominations

| Ticker | Currency |
|---|---|
| Stock_01 | Crncy_03 |
| Stock_02 | Crncy_04 |
| Stock_03 | Crncy_04 |
| Stock_04 | Crncy_02 |
| Stock_05 | Crncy_03 |
| Stock_06 | Crncy_02 |
| Stock_07 | Crncy_03 |
| Stock_08 | Crncy_02 |
| Stock_09 | Crncy_04 |
| Stock_10 | Crncy_03 |
| Stock_11 | Crncy_01 |
| Stock_12 | Crncy_04 |
| Stock_13 | Crncy_01 |
| Stock_14 | Crncy_01 |
| Stock_15 | Crncy_01 |

### FX pairs

| Ticker | Pair |
|---|---|
| FX_01 | Crncy_02 / Crncy_01 |
| FX_02 | Crncy_04 / Crncy_02 |
| FX_03 | Crncy_04 / Crncy_03 |
| FX_04 | Crncy_02 / Crncy_03 |
| FX_05 | Crncy_01 / Crncy_03 |
| FX_06 | Crncy_04 / Crncy_01 |

### Special instruments

- **`Idx_04`** is an ETF that is equally weighted across all equities, with **no rebalancing** after inception.

## Tips

- Start with the notebook (`Quickstart.ipynb`) to understand the data flow
- Model and analyze the data for inspiration
- Use the visualization to identify when your strategy performs well/poorly
- Keep `trading_simulator.py` unchanged for consistency across backtesting and evaluation

Good luck building!