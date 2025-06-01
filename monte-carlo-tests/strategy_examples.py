#########################################
#                                       #
#  EXAMPLE STRATEGIES FOR BACKTESTING   #
#                                       #
#########################################


# ---------------------------- IMPORTS ---------------------------- #

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt


# ---------------------------- DONCHIAN BREAKOUT STRATEGY ---------------------------- #


def donchian_breakout(returns: pd.DataFrame, lookback: int):
    """
    Donchian Breakout-based strategy with custom lookback period/window.
    This approach relies on a high-low price channel.
    Args:
        | returns (pd.DataFrame): Historical price data for a ticker.
        | lookback (int): lookback period for calculating upper/lower channel limits.
    """
    # Assumes that returns dataframe contains a "Close" column.
    upper_limit = returns["Close"].rolling(lookback - 1).max().shift(1)
    lower_limit = returns["Close"].rolling(lookback - 1).min().shift(1)
    signal = pd.Series(np.full(len(returns), np.nan), index=returns.index)
    signal.loc[returns["Close"] > upper_limit] = 1
    signal.loc[returns["Close"] < lower_limit] = -1
    signal = signal.ffill()
    return signal


def optimise_donchian(returns: pd.DataFrame):
    """
    Optimise Donchian Breakout strategy to find optimal lookback.
    Args:
        | returns (pd.DataFrame): Historical price data for a ticker.
    """
    best_profit_factor = 0
    best_lookback = -1
    r = np.log(returns["Close"]).diff().shift(-1)
    for i in range(5, 365):
        signal = donchian_breakout(returns=returns, lookback=i)
        signal_returns = r * signal
        profit_factor = (
            signal_returns[signal_returns > 0].sum()
            / signal_returns[signal_returns < 0].abs().sum()
        )
        if profit_factor > best_profit_factor:
            best_profit_factor = profit_factor
            best_lookback = i
    return best_lookback, best_profit_factor


# ---------------------------- SIMPLE BACKTEST ---------------------------- #

if __name__ == "__main__":
    ticker = "GBPUSD=X"
    start_date = "2016-01-01"
    end_date = "2021-01-01"
    returns_df = yf.download(ticker, start=start_date, end=end_date)
    returns_df.columns = returns_df.columns.get_level_values(0)
    best_lookback, best_profit_factor = optimise_donchian(returns_df)
    print(f"Best lookback = {best_lookback} days.")
    print(f"Best profit factor = {best_profit_factor}.")
    signal = donchian_breakout(returns_df, best_lookback)
    returns_df["returns"] = np.log(returns_df["Close"]).diff().shift(-1)
    returns_df["donchian_returns"] = returns_df["returns"] * signal
    plt.style.use("dark_background")
    returns_df["donchian_returns"].cumsum().plot(color="red")
    plt.title("In-Sample Donchian Breakout")
    plt.ylabel("Cumulative Log Return")
    plt.show()
