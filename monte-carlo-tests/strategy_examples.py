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


# ---------------------------- MOVING AVERAGE CROSSOVER STRATEGY ---------------------------- #


def moving_average_crossover(returns: pd.DataFrame, short: int, long: int) -> pd.Series:
    """
    Moving average crossover strategy.
    This approach uses a short-term average and a long-term average, entering into
    a buy position if the short-term average crosses above the long-term average,
    and into a short position if the opposite occurs.
    Args:
        | returns (pd.DataFrame): Historical price data for a ticker.
        | short (int): Duration in days used for short-term average.
        | short (int): Duration in days used for long-term average.
    Returns:
        | signal (pd.Series): Array representing position changes over time.
    """
    short_avg = returns["Close"].rolling(short).mean()
    long_avg = returns["Close"].rolling(long).mean()
    signal = pd.Series(np.full(len(returns), np.nan), index=returns.index)
    signal[short_avg > long_avg] = 1
    signal[short_avg < long_avg] = -1
    return signal.ffill()


# ---------------------------- MEAN REVERSION STRATEGY ---------------------------- #


def mean_reversion_zscore(returns: pd.DataFrame, lookback: int) -> pd.Series:
    """
    Mean reversion with zscore strategy.
    This approach assumes that in the long term, the price of a security will
    revert to its mean.
    Args:
        | returns (pd.DataFrame): Historical price data for a ticker.
        | lookback (int): Period in days used for computing mean.
    Returns:
        | signal (pd.Series): Array representing position changes over time.
    """
    rolling_mean = returns["Close"].rolling(lookback).mean()
    rolling_std = returns["Close"].rolling(lookback).std()
    z = (returns["Close"] - rolling_mean) / rolling_std
    signal = pd.Series(0, index=returns.index)
    signal[z > 1] = -1
    signal[z < -1] = 1
    return signal.ffill()


# ---------------------------- DONCHIAN BREAKOUT STRATEGY ---------------------------- #


def donchian_breakout(returns: pd.DataFrame, lookback: int) -> pd.Series:
    """
    Donchian Breakout-based strategy with custom lookback period/window.
    This approach relies on a high-low price channel.
    Args:
        | returns (pd.DataFrame): Historical price data for a ticker.
        | lookback (int): lookback period for calculating upper/lower channel limits.
    Returns:
        | signal (pd.Series): Array representing position changes over time.
    """
    # Assumes that returns dataframe contains a "Close" column.
    upper_limit = returns["Close"].rolling(lookback - 1).max().shift(1)
    lower_limit = returns["Close"].rolling(lookback - 1).min().shift(1)
    signal = pd.Series(np.full(len(returns), np.nan), index=returns.index)
    signal.loc[returns["Close"] > upper_limit] = 1
    signal.loc[returns["Close"] < lower_limit] = -1
    return signal.ffill()


def optimise_donchian(returns: pd.DataFrame) -> tuple[int, int]:
    """
    Optimise Donchian Breakout strategy to find optimal lookback.
    Args:
        | returns (pd.DataFrame): Historical price data for a ticker.
    Returns:
        | best_lookback (int): Optimised lookback period in days.
        | best_profit_factor (float): Profit factor for optimised lookback.
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
