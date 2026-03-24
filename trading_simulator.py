import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

# ===== SIMULATE TRADING AND GENERATE ORDERS =====
class TradingSimulator:
    def __init__(self, assets, initial_cash=100_000, equity_currency_map=None, fx_pairs_map=None):
        self.assets = assets
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.portfolio = {ticker: 0 for ticker in assets}
        self.orders = []
        self.portfolio_snapshots = []
        self.equity_currency_map = equity_currency_map or {}
        self.fx_pairs_map = fx_pairs_map or {}

    def execute_order(self, date, ticker, action, shares, price):
        """Execute a buy/sell order and record it.

        Shorting:
          SELL with holdings >= 0  → opens a short position (holdings goes negative)
          BUY  with holdings < 0   → covers a short (partial or full)

        There is no cash requirement for opening a short, but the short
        position contributes negative value to the portfolio.
        """
        action = action.upper()
        if shares <= 0:
            return False

        if action == 'BUY':
            cost = price * shares
            if cost > self.cash:
                return False
            self.cash -= cost
            self.portfolio[ticker] += shares
            self.orders.append({
                'Date': date,
                'Ticker': ticker,
                'Action': 'BUY',
                'Shares': shares,
                'Price': price,
                'Total': -cost,
            })
            return True

        if action == 'SELL':
            proceeds = price * shares
            self.cash += proceeds
            self.portfolio[ticker] -= shares
            self.orders.append({
                'Date': date,
                'Ticker': ticker,
                'Action': 'SELL',
                'Shares': shares,
                'Price': price,
                'Total': proceeds,
            })
            return True

        return False

    def run(self, strategy_fn, prices_df, data):
        """Run a strategy function that returns a list of orders per day."""
        for row_pos, (idx, row) in enumerate(prices_df.iterrows()):
            date = idx
            signal_prices = {ticker: row[ticker] for ticker in self.assets}
            orders = strategy_fn(
                row_pos,       # integer row index for iloc-based slicing
                self.cash,
                self.portfolio.copy(),
                signal_prices,
                data,
            ) or []

            if isinstance(orders, tuple):
                orders = [orders]
            
            # Execute at NEXT day's closing price -- 1-day execution lag
            if row_pos + 1 < len(prices_df):
                next_row    = prices_df.iloc[row_pos + 1]
                exec_prices = {ticker: next_row[ticker] for ticker in self.assets}
                exec_date   = prices_df.index[row_pos + 1]
            else:
                exec_prices = signal_prices   # last day: no next row available
                exec_date   = date

            for action, ticker, shares in orders:
                self.execute_order(exec_date, ticker, action, shares, exec_prices[ticker])

            self.record_portfolio(date, signal_prices)

    def execute_trade(self, date, ticker, signal, price, shares=10):
        """Convenience wrapper around execute_order using numeric signals.

        signal =  1 → BUY
        signal = -1 → SELL (or open/extend short)
        """
        if signal == 1:
            return self.execute_order(date, ticker, 'BUY', shares, price)
        elif signal == -1:
            return self.execute_order(date, ticker, 'SELL', shares, price)
        return False

    def record_portfolio(self, date, prices_row):
        """Record portfolio snapshot including FX exposure per currency."""
        snapshot = {'Date': date, 'Cash': self.cash}

        for ticker in self.assets:
            shares = self.portfolio[ticker]
            snapshot[f'{ticker}_Shares'] = shares
            snapshot[f'{ticker}_Value'] = shares * prices_row[ticker]

        total_holdings = sum(
            self.portfolio[ticker] * prices_row[ticker] for ticker in self.assets
        )
        snapshot['Total_Value'] = self.cash + total_holdings

        # Net FX exposure per currency:
        #   equity leg:  sum(holdings * price) per currency
        #   FX hedge leg: long FX(A/B) → +notional to A, -notional to B
        #                 short FX(A/B) → -notional to A, +notional to B
        # Net ≈ 0 means fully hedged.
        if self.equity_currency_map:
            net: dict[str, float] = {}

            # Equity leg
            for asset, crncy in self.equity_currency_map.items():
                val = self.portfolio.get(asset, 0) * prices_row.get(asset, 0)
                net[crncy] = net.get(crncy, 0.0) + val

            # FX hedge leg
            for fx_ticker, (base, quote) in self.fx_pairs_map.items():
                notional = self.portfolio.get(fx_ticker, 0) * prices_row.get(fx_ticker, 0)
                net[base]  = net.get(base,  0.0) + notional   # long base
                net[quote] = net.get(quote, 0.0) - notional   # short quote

            # Hedge score: fraction of FX exposure offset by FX hedges (0-100).
            # 0 = unhedged, 100 = fully hedged, negative = over-hedged.
            # Baseline is equity-only exposure so stock-only strategies score 0
            # and commodity/index-only strategies (nothing in equity_currency_map)
            # score 100 but are flagged as N/A in displays.
            unhedged: dict[str, float] = {}
            for asset, crncy in self.equity_currency_map.items():
                val = self.portfolio.get(asset, 0) * prices_row.get(asset, 0)
                unhedged[crncy] = unhedged.get(crncy, 0.0) + val

            rms_unhedged = float(np.sqrt(np.mean([v**2 for v in unhedged.values()]))) if unhedged else 0.0
            rms_hedged   = float(np.sqrt(np.mean([v**2 for v in net.values()])))      if net      else 0.0
            hedge_score  = float(max(0.0, 100 * (1 - rms_hedged / max(rms_unhedged, 1.0))))

            snapshot['_fx_exposure']   = net
            snapshot['_hedge_score']   = hedge_score
            snapshot['_rms_unhedged']  = rms_unhedged

        self.portfolio_snapshots.append(snapshot)

    def save_results(self, orders_file='orders.csv', portfolio_file='portfolio.csv'):
        """Save orders and portfolio to CSV files"""
        # Save orders
        orders_df = pd.DataFrame(self.orders)
        orders_df.to_csv(orders_file, index=False)
        print(f"Saved {len(orders_df)} orders to '{orders_file}'")

        # Save portfolio
        portfolio_df = pd.DataFrame(self.portfolio_snapshots)
        portfolio_df.to_csv(portfolio_file, index=False)
        print(f"Saved {len(portfolio_df)} portfolio snapshots to '{portfolio_file}'")

    def calculate_sharpe_ratio(self):
        portfolio_df = pd.DataFrame(self.portfolio_snapshots)
        daily_pnl = portfolio_df['Total_Value'].diff().dropna()

        if len(daily_pnl) < 2 or daily_pnl.std() == 0:
            return 0.0

        return float(np.sqrt(260) * daily_pnl.mean() / daily_pnl.std())
    
    def calculate_metrics(self):
        """
        Compute full performance metrics from portfolio history.

        Returns
        -------
        dict with keys:
            adjusted_sharpe, sharpe, sharpe_1d_lag, skewness,
            max_drawdown, nbr_of_trades,
            returns (Series), rolling_sharpe (Series), rolling_std (Series),
            fx_hedge_scores (list)
        """
        portfolio_df = pd.DataFrame(self.portfolio_snapshots).set_index('Date')
        portfolio_df.index = pd.to_datetime(portfolio_df.index)
        total_val    = portfolio_df['Total_Value']
        daily_pnl    = total_val.diff().dropna()

        if len(daily_pnl) < 2 or daily_pnl.std() == 0:
            return {}

        ann    = np.sqrt(260)
        sharpe = float(ann * daily_pnl.mean() / daily_pnl.std())
        skew   = float(daily_pnl.skew())
        kurt   = float(daily_pnl.kurt())   # excess kurtosis

        # Adjusted Sharpe -- penalty for skew & fat tails
        adj_sharpe = sharpe * (1 + sharpe / 6 * skew - (sharpe ** 2) / 24 * kurt)

        # Sharpe 1d lag -- large gap vs sharpe suggests lookahead bias
        lagged     = daily_pnl.shift(1).dropna()
        sharpe_lag = float(ann * lagged.mean() / lagged.std()) if lagged.std() > 0 else 0.0

        # Max drawdown
        cum    = daily_pnl.cumsum()
        max_dd = float((cum - cum.cummax()).min())


        # Rolling metrics (1-year window)
        w = 260
        rolling_sharpe = (
            daily_pnl.rolling(w, min_periods=w // 2).mean()
            / daily_pnl.rolling(w, min_periods=w // 2).std()
            * ann
        )
        daily_ret_pct = (total_val.pct_change().dropna() * 100).reindex(daily_pnl.index)
        rolling_std = daily_ret_pct.rolling(w, min_periods=w // 2).std()

        # FX hedge scores
        fx_hedge_scores = [
            s['_hedge_score']
            for s in self.portfolio_snapshots
            if '_hedge_score' in s
        ]
        if fx_hedge_scores:
            snap_dates_fx = [s['Date'] for s in self.portfolio_snapshots if '_hedge_score' in s]
            fx_hedge_score_series = pd.Series(fx_hedge_scores, index=pd.to_datetime(snap_dates_fx))
        else:
            fx_hedge_score_series = pd.Series(dtype=float)

        return dict(
            adjusted_sharpe       = adj_sharpe,
            sharpe                = sharpe,
            sharpe_1d_lag         = sharpe_lag,
            skewness              = skew,
            max_drawdown          = max_dd,
            nbr_of_trades         = len(self.orders),
            returns               = daily_pnl,
            rolling_sharpe        = rolling_sharpe,
            rolling_std           = rolling_std,
            fx_hedge_scores       = fx_hedge_scores,
            fx_hedge_score_series = fx_hedge_score_series,
        )


    def plot_performance(self, prices_df, save_file='performance_plot.png'):
        """
        Render a full performance dashboard.

        Layout
        ------
        Row 0 : Cumulative P&L
        Row 1 : Buy/sell trade-count bars
        Row 2 : Rolling group (Sharpe / FX Hedge / Volatility) | Pie chart + Key figures table
        """
        BG     = '#0b1120';  AX_BG  = '#111827';  GRID   = '#1e2d45'
        TEXT   = '#c9d4e8';  BLUE   = '#38bdf8';  GREEN  = '#10b981'
        RED    = '#f43f5e';  INDIGO = '#818cf8';  AMBER  = '#fbbf24'
        PINK   = '#f472b6'

        GROUP_COLORS = {'Stock': '#38bdf8', 'Comm': '#818cf8', 'Idx': '#a78bfa', 'FX': '#c084fc'}

        def style_ax(ax, title='', xlabel='', ylabel='', hide_xticks=False):
            ax.set_facecolor(AX_BG)
            if title:
                ax.set_title(title, color=TEXT, fontsize=10, fontweight='bold', pad=6)
            if xlabel:
                ax.set_xlabel(xlabel, color=TEXT, fontsize=9)
            ax.set_ylabel(ylabel, color=TEXT, fontsize=9)
            ax.tick_params(colors=TEXT, labelsize=8)
            ax.grid(color=GRID, linewidth=0.5, alpha=0.8)
            for sp in ax.spines.values():
                sp.set_edgecolor(GRID)
            if hide_xticks:
                plt.setp(ax.get_xticklabels(), visible=False)

        # Compute metrics
        m            = self.calculate_metrics()
        portfolio_df = pd.DataFrame(self.portfolio_snapshots)
        dates        = pd.to_datetime(prices_df.index)
        total_val    = portfolio_df['Total_Value']
        cum_pnl_pct  = (total_val - self.initial_cash) / self.initial_cash * 100
        final_value  = total_val.iloc[-1]
        net_profit   = final_value - self.initial_cash
        net_pct      = net_profit / self.initial_cash * 100
        fx_scores    = m.get('fx_hedge_scores', [])

        with plt.style.context('dark_background'):
            fig = plt.figure(figsize=(16, 9), facecolor=BG)

        outer = fig.add_gridspec(2, 1, height_ratios=[3.2, 5],
                                 hspace=0.22, left=0.05, right=0.97,
                                 top=0.92, bottom=0.05)
        top_gs     = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0],
                                             height_ratios=[2.2, 1.0], hspace=0.08)
        row2       = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[1], wspace=0.15,
                                             width_ratios=[1.18, 1])
        rolling_gs = GridSpecFromSubplotSpec(3, 1, subplot_spec=row2[0], hspace=0.15)
        right_gs   = GridSpecFromSubplotSpec(1, 2, subplot_spec=row2[1], wspace=0.25)

        # Row 0: cumulative P&L
        ax_pnl = fig.add_subplot(top_gs[0])
        ax_pnl.plot(dates, cum_pnl_pct, color=BLUE, lw=1.5)
        ax_pnl.fill_between(dates, 0, cum_pnl_pct,
                            where=(cum_pnl_pct >= 0), color=GREEN, alpha=0.20, interpolate=True)
        ax_pnl.fill_between(dates, 0, cum_pnl_pct,
                            where=(cum_pnl_pct < 0), color=RED, alpha=0.20, interpolate=True)
        ax_pnl.axhline(0, color=TEXT, lw=0.8, alpha=0.5)
        style_ax(ax_pnl, f'Cumulative P&L  .  ${net_profit:+,.0f}  ({net_pct:+.2f}%)',
                 ylabel='P&L (%)', hide_xticks=True)

        # Row 1: trade bars
        ax_tr = fig.add_subplot(top_gs[1], sharex=ax_pnl)
        if self.orders:
            odf = pd.DataFrame(self.orders)
            odf['Date'] = pd.to_datetime(odf['Date'])
            buys  = odf[odf['Action'] == 'BUY'].groupby('Date').size()
            sells = odf[odf['Action'] == 'SELL'].groupby('Date').size()
            ax_tr.bar(buys.index,   buys.values,  color=GREEN, alpha=0.8, width=1.2, label='Buys')
            ax_tr.bar(sells.index, -sells.values, color=RED,   alpha=0.8, width=1.2, label='Sells')
            ax_tr.axhline(0, color=TEXT, lw=0.5, alpha=0.4)
            ax_tr.legend(loc='upper right', fontsize=7, framealpha=0.2,
                         facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT, ncol=2)
        style_ax(ax_tr, '', xlabel='Date', ylabel='# Trades')

        # Row 2 left: rolling group (3 stacked subplots)
        # Rolling Sharpe
        ax_rs = fig.add_subplot(rolling_gs[0])
        rs = m.get('rolling_sharpe', pd.Series(dtype=float))
        ax_rs.plot(rs.index, rs, color=INDIGO, lw=1.4)
        ax_rs.fill_between(rs.index, rs, 0, where=(rs < 0),
                           color=RED, alpha=0.25, interpolate=True)
        ax_rs.axhline(0,  color=TEXT,  lw=0.8, ls='--', alpha=0.4)
        ax_rs.axhline(1,  color=GREEN, lw=0.8, ls='--', alpha=0.5, label='SR = 1')
        ax_rs.axhline(-1, color=RED,   lw=0.8, ls='--', alpha=0.4)
        ax_rs.legend(fontsize=8, framealpha=0.2, facecolor=AX_BG,
                     edgecolor=GRID, labelcolor=TEXT)
        style_ax(ax_rs, 'Rolling window (1y)', ylabel='Sharpe', hide_xticks=True)

        # FX Hedge Score
        ax_fx = fig.add_subplot(rolling_gs[1], sharex=ax_rs)
        fx_series = m.get('fx_hedge_score_series', pd.Series(dtype=float))
        if len(fx_series) > 0:
            ax_fx.plot(fx_series.index, fx_series, color=INDIGO, lw=1.4)
            ax_fx.axhline(100, color=GREEN, lw=0.8, ls='--', alpha=0.5, label='Perfect (100)')
            ax_fx.set_ylim(0, 105)
            ax_fx.legend(fontsize=8, framealpha=0.2, facecolor=AX_BG,
                         edgecolor=GRID, labelcolor=TEXT)
        style_ax(ax_fx, '', ylabel='FX Hedge', hide_xticks=True)

        # Rolling Volatility
        ax_vol = fig.add_subplot(rolling_gs[2], sharex=ax_rs)
        std = m.get('rolling_std', pd.Series(dtype=float))
        ax_vol.plot(std.index, std, color=INDIGO, lw=1.4)
        ax_vol.fill_between(std.index, std, 0, color=INDIGO, alpha=0.15)
        style_ax(ax_vol, '', ylabel='Volatility (%)', xlabel='Date')
        ax_vol.xaxis.set_label_coords(0.5, -0.15)

        # Row 2 right left: asset class allocation pie (by total traded volume)
        ax_pie = fig.add_subplot(right_gs[0])
        ax_pie.set_facecolor(AX_BG)
        traded_tickers = set(o['Ticker'] for o in self.orders) if self.orders else set()
        n_traded = len(traded_tickers)
        n_total  = len(self.assets)
        group_vals = {}
        if self.orders:
            odf_v = pd.DataFrame(self.orders)
            odf_v['Volume'] = odf_v['Price'] * odf_v['Shares'].abs()
            for _, row in odf_v.iterrows():
                grp = next((p for p in GROUP_COLORS if row['Ticker'].startswith(p)), 'Other')
                group_vals[grp] = group_vals.get(grp, 0) + row['Volume']
        if group_vals:
            pie_labels = list(group_vals.keys())
            pie_sizes  = list(group_vals.values())
            pie_colors = [GROUP_COLORS.get(lb, BLUE) for lb in pie_labels]
            _, _, autotexts = ax_pie.pie(
                pie_sizes, labels=pie_labels, colors=pie_colors,
                autopct='%1.1f%%', startangle=90,
                textprops={'color': TEXT, 'fontsize': 8},
                wedgeprops={'edgecolor': AX_BG, 'linewidth': 1.5},
                radius=1.0,
            )
            for at in autotexts:
                at.set_color(TEXT)
                at.set_fontsize(8)
        else:
            ax_pie.text(0.5, 0.5, 'No trades',
                        ha='center', va='center', color=TEXT, fontsize=9,
                        transform=ax_pie.transAxes)
        ax_pie.set_title('Asset Allocation (Traded Volume)',
                         color=TEXT, fontsize=8, fontweight='bold', pad=4)
        ax_pie.text(0.5, -0.04, f'Traded assets   {n_traded}/{n_total}',
                    ha='center', va='top', color=TEXT, fontsize=8,
                    transform=ax_pie.transAxes)

        # Row 2 right right: key figures table
        ax_tbl = fig.add_subplot(right_gs[1])
        ax_tbl.axis('off')
        fx_score_val = float(np.mean(fx_scores)) if fx_scores else None
        has_fx_exposure = any(
            s.get('_rms_unhedged', 0) > 0
            for s in self.portfolio_snapshots
        )
        WHITE_ROWS = {6, 7}  # FX Hedge Score, # Trades always white
        rows_data = [
            ["Net P&L",           f'${net_profit:+,.0f}'],
            ["Sharpe",            f'{m.get("sharpe", 0):+.3f}'],
            ["Adj. Sharpe",       f'{m.get("adjusted_sharpe", 0):+.3f}'],
            ["Sharpe 1d lag",     f'{m.get("sharpe_1d_lag", 0):+.3f}'],
            ["Skewness",          f'{m.get("skewness", 0):+.3f}'],
            ["Max DD",            f'${m.get("max_drawdown", 0):+,.0f}'],
            ["FX Hedge Score",    f'{fx_score_val:.1f}/100' if (fx_score_val is not None and has_fx_exposure) else 'N/A'],
            ["# Trades",          f'{m.get("nbr_of_trades", 0):,}'],
        ]
        tbl = ax_tbl.table(cellText=rows_data, loc='upper center',
                           bbox=[0.0, 0.0, 1.0, 1.0], edges='open')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_text_props(color=TEXT)
            if c == 1 and r < len(rows_data):
                if r in WHITE_ROWS:
                    cell.set_text_props(color=TEXT, fontweight='bold')
                else:
                    try:
                        val = float(rows_data[r][1].replace('%','').replace('$','')
                                    .replace(',','').split()[0])
                        cell.set_text_props(
                            color=GREEN if val > 0 else (RED if val < 0 else TEXT),
                            fontweight='bold')
                    except (ValueError, IndexError):
                        pass
        ax_tbl.set_title('Key Figures', color=TEXT, fontsize=10, fontweight='bold', pad=6)

        # Suptitle
        fig.suptitle(
            'LINC Hackathon 2026',
            color=TEXT, fontsize=10, fontweight='bold', y=0.97,
        )

        plt.savefig(save_file, dpi=150, bbox_inches='tight', facecolor=BG)
        print(f"Saved performance plot to '{save_file}'")
        plt.show()
        plt.close()

        # Text summary
        w = 52
        print(f"\n{'='*w}")
        print("  TRADING SUMMARY")
        print(f"{'='*w}")
        print(f"  Initial Cash      ${self.initial_cash:>14,.2f}")
        print(f"  Final Value       ${final_value:>14,.2f}")
        print(f"  Net P&L           ${net_profit:>+14,.2f}  ({net_pct:+.2f}%)")
        print(f"  Sharpe            {m.get('sharpe', 0):>14.3f}")
        print(f"  Adjusted Sharpe   {m.get('adjusted_sharpe', 0):>14.3f}")
        print(f"  Sharpe 1d lag     {m.get('sharpe_1d_lag', 0):>14.3f}")
        print(f"  Skewness          {m.get('skewness', 0):>14.3f}")
        if fx_scores:
            print(f"  FX Hedge Score    {fx_scores[-1]:>13.1f}/100")
        print(f"  Max Drawdown      ${m.get('max_drawdown', 0):>+14,.2f}")
        print(f"  # Trades          {m.get('nbr_of_trades', 0):>14,}")
        print(f"{'='*w}\n")