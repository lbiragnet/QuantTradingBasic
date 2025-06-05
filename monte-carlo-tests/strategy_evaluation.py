#########################################
#                                       #
# STRATEGY EVALUATION WITH PERMUTATIONS #
#                                       #
#########################################


# ---------------------------- IMPORTS ---------------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategy_examples import *
from typing import List, Union
from joblib import Parallel, delayed


# ---------------------------- IN-SAMPLE INITIAL TEST ---------------------------- #


def evaluate_strategy(signal: pd.Series, returns: pd.DataFrame) -> tuple[float, float]:
    """
    Evaluate a strategy based on some given time series data.
    Args:
        | signal (pd.Series): Vector describing positions taken by the strategy through time.
        | returns (pd.DataFrame): Historical price data for a ticker.
    Returns:
        | pf (float): Profit factor for the strategy and data.
        | sharpe (float): Sharpe ratio for the strategy and data.
    """
    strat_returns = signal * returns
    pf = (
        strat_returns[strat_returns > 0].sum()
        / strat_returns[strat_returns < 0].abs().sum()
    )
    sharpe = strat_returns.mean() / strat_returns.std() * np.sqrt(252)
    return pf, sharpe


def run_backtest(returns: pd.DataFrame, strategy_name: str, param_grid):
    """
    First step in strategy testing - run an initial test on in-sample data.
    This function optimises the strategy based on some given price time series.
    Args:
        | returns (pd.DataFrame): Historical price data for a ticker.
        | strategy_name (str): The name of the strategy.
        | param_grid: Gride of parameters to be used with the strategy.
    Returns:
        | best_params: Best parameters found by optimising the strategy.
        | best_pf (float): Best profit factor found by optimising the strategy.
        | best_signal (pd.Series): Vector describing positions taken by the strategy through time.
    """
    log_returns = np.log(returns["Close"]).diff().shift(-1)
    best_pf = -np.inf
    best_params = None
    best_signal = None
    for params in param_grid:
        if strategy_name == "donchian":
            signal = donchian_breakout(returns, **params)
        elif strategy_name == "ma_cross":
            signal = moving_average_crossover(returns, **params)
        elif strategy_name == "mean_revert":
            signal = mean_reversion_zscore(returns, **params)
        else:
            raise ValueError("Unknown strategy")
        pf, sharpe = evaluate_strategy(signal, log_returns)
        if pf > best_pf:
            best_pf = pf
            best_params = params
            best_signal = signal
    return best_params, best_pf, best_signal


# ---------------------------- IN-SAMPLE PERMUTATION TEST ---------------------------- #


def get_permutation(
    returns: Union[pd.DataFrame, List[pd.DataFrame]], start_index: int = 0, seed=None
):
    """
    This function produce a permutation of some given price time series.
    It maintains statistical characteristics (mean, skew, kurtosis, etc.)
    It also maintains correlation if using multiple assets.
    Args:
        | returns ([pd.DataFrame] | pd.DataFrame): Historical price data for one or more tickers.
        | start_index (int): Index to start permuting.
        | seed: Seed for random selection.
    Returns:
        | perm_returns ([pd.DataFrame] | pd.DataFrame): Permuted price data.
    """
    assert start_index >= 0
    np.random.seed(seed)
    # Allow handling of both single and multiple price time series
    if isinstance(returns, list):
        time_index = returns[0].index
        for market in returns:
            assert np.all(time_index == market.index), "Indexes do not match"
        n_markets = len(returns)
    else:
        n_markets = 1
        time_index = returns.index
        returns = [returns]
    # Setup data structures
    n_bars = len(returns[0])
    perm_index = start_index + 1
    perm_n = n_bars - perm_index
    start_bar = np.empty((n_markets, 4))
    relative_open = np.empty((n_markets, perm_n))
    relative_high = np.empty((n_markets, perm_n))
    relative_low = np.empty((n_markets, perm_n))
    relative_close = np.empty((n_markets, perm_n))
    for market_i, reg_bars in enumerate(returns):
        # Log-transform prices to work in log-return space
        log_bars = np.log(reg_bars[["Open", "High", "Low", "Close"]])
        start_bar[market_i] = log_bars.iloc[start_index].to_numpy()
        r_o = (log_bars["Open"] - log_bars["Close"].shift()).to_numpy()
        r_h = (log_bars["High"] - log_bars["Open"]).to_numpy()
        r_l = (log_bars["Low"] - log_bars["Open"]).to_numpy()
        r_c = (log_bars["Close"] - log_bars["Open"]).to_numpy()
        relative_open[market_i] = r_o[perm_index:]
        relative_high[market_i] = r_h[perm_index:]
        relative_low[market_i] = r_l[perm_index:]
        relative_close[market_i] = r_c[perm_index:]
    # Shuffle bars for relative values
    idx = np.arange(perm_n)
    perm1 = np.random.permutation(idx)
    relative_high = relative_high[:, perm1]
    relative_low = relative_low[:, perm1]
    relative_close = relative_close[:, perm1]
    perm2 = np.random.permutation(idx)
    # Close to open gap is shuffled separately
    relative_open = relative_open[:, perm2]
    # Create permutation from relative prices
    perm_returns = []
    for market_i, reg_bars in enumerate(returns):
        perm_bars = np.zeros((n_bars, 4))
        # Copy over real data before start index
        log_bars = np.log(reg_bars[["Open", "High", "Low", "Close"]]).to_numpy().copy()
        perm_bars[:start_index] = log_bars[:start_index]
        # Copy start bar
        perm_bars[start_index] = start_bar[market_i]
        for i in range(perm_index, n_bars):
            k = i - perm_index
            perm_bars[i, 0] = perm_bars[i - 1, 3] + relative_open[market_i][k]
            perm_bars[i, 1] = perm_bars[i, 0] + relative_high[market_i][k]
            perm_bars[i, 2] = perm_bars[i, 0] + relative_low[market_i][k]
            perm_bars[i, 3] = perm_bars[i, 0] + relative_close[market_i][k]
        perm_bars = np.exp(perm_bars)
        perm_bars = pd.DataFrame(
            perm_bars, index=time_index, columns=["Open", "High", "Low", "Close"]
        )
        perm_returns.append(perm_bars)
    if n_markets > 1:
        return perm_returns
    else:
        return perm_returns[0]


def permutation_test(
    returns: pd.DataFrame,
    strategy_name: str,
    param_grid,
    n_permutations: int = 1000,
    start_index: int = 0,
    seed=None,
):
    """
    Second step in strategy testing - run an in-sample permutation test.
    It is used for assessing data mining bias/overfitting, and optimises a strategy using real price data
    and permuted price time series.
    It produces two figures:
        1) A plot of log returns on real data for the strategy optimised on real in-sample data,
           and of log returns on real data for the stragies optimised on permuted in-sample data.
        2) A histogram of profit factors obtained on real and permuted in-sample data. Aim for <1% p-value.
    Args:
        | returns (pd.DataFrame): Historical price data a ticker.
        | strategy_name (str): Name of the strategy.
        | param_grid: Grid of parameters for the strategy.
        | n_permutations (int): Number of permutations to use (recommended is 1000).
        | start_index (int): Index to start permutation.
        | seed: Seed for random selection.
    """
    log_returns = np.log(returns["Close"]).diff()
    _, pf_real, signal_real = run_backtest(returns, strategy_name, param_grid)
    plt.figure()
    print(f"In-sample PF = {pf_real}.")
    perm_better_count = 1
    permuted_profit_factors = []
    for _ in range(1, n_permutations):
        train_perm = get_permutation(returns, start_index, seed)
        _, pf_perm, signal_perm = run_backtest(train_perm, strategy_name, param_grid)
        # Plot returns for permuted-trained strategy
        signal_perm = signal_perm.reindex(returns.index).fillna(0)
        valid_mask = (~log_returns.isna()) & (
            ~signal_perm.shift().isna()
        )  # Shift to common index
        strategy_returns = (log_returns * signal_perm.shift())[valid_mask]
        strategy_returns.cumsum().plot(color="lightgrey", alpha=0.3)
        if pf_perm >= pf_real:
            perm_better_count += 1
        permuted_profit_factors.append(pf_perm)
    # Plot returns for real-trained strategy
    signal_real = signal_real.reindex(returns.index).fillna(0)
    valid_mask = (~log_returns.isna()) & (
        ~signal_real.shift().isna()
    )  # Shift to common index
    strategy_returns = (log_returns * signal_real.shift())[valid_mask]
    strategy_returns.cumsum().plot(color="red", label="In-sample real", alpha=1)
    insample_perm_pval = perm_better_count / n_permutations
    print(f"In-sample Monte Carlo permutations P-Value = {insample_perm_pval}.")
    plt.title("In-Sample Monte Carlo Permutation Test: " + strategy_name)
    plt.ylabel("Cumulative Log Return")
    plt.show()
    pd.Series(permuted_profit_factors).hist(color="blue", label="Permutations")
    plt.axvline(pf_real, color="red", label="Real")
    plt.xlabel("Profit Factor")
    plt.title(f"In-sample MCPT P-Value = {insample_perm_pval}")
    plt.grid(False)
    plt.legend()
    plt.show()


# ---------------------------- WALKFORWARD TEST ---------------------------- #


def walkforward_test(
    returns: pd.DataFrame,
    strategy_name: str,
    param_grid,
    train_lookback: int = 252 * 4,  # 4 years of daily data
    train_step: int = 21,  # Step forward by ~1 month
) -> pd.Series:
    """
    Third step in strategy testing - run a walkforward test.
    Effectively uses out-of-sample data for evaluating a strategy, optimising it based on
    a time window and then using this to perform trades in the next time window.
    Args:
        | returns (pd.DataFrame): Historical price data.
        | strategy_name (str): Strategy name ('donchian', 'ma_cross', etc.).
        | param_grid: Parameter grid for optimization.
        | train_lookback (int): Size of training window (in rows).
        | train_step (int): Step size for re-optimization.
    Returns:
        | wf_signal (pd.Series): Strategy signal generated via walkforward testing.
    """
    log_returns = np.log(returns["Close"]).diff().shift(-1)
    n = len(returns)
    wf_signal = pd.Series(index=returns.index, dtype=float)
    next_train = train_lookback
    while next_train < n:
        # Train slice
        train_data = returns.iloc[next_train - train_lookback : next_train]
        # Optimize on training window
        best_params, _, _ = run_backtest(train_data, strategy_name, param_grid)
        # Apply to current point (just the next day)
        if strategy_name == "donchian":
            signal = donchian_breakout(returns, **best_params)
        elif strategy_name == "ma_cross":
            signal = moving_average_crossover(returns, **best_params)
        elif strategy_name == "mean_revert":
            signal = mean_reversion_zscore(returns, **best_params)
        else:
            raise ValueError("Unknown strategy")
        start_idx = next_train
        end_idx = min(next_train + train_step, n)
        # Apply best parameters to full data, and slice next segment
        full_signal = signal
        wf_signal.iloc[start_idx:end_idx] = full_signal.iloc[start_idx:end_idx]
        next_train += train_step
    wf_signal = wf_signal.ffill()
    return wf_signal


# ---------------------------- WALKFORWARD PERMUTATION TEST ---------------------------- #


def single_wf_permutation_test(
    returns, strategy_name, param_grid, train_lookback, train_step, seed
):
    """
    Building block for walkforward permutation test - conduct one permutation test
    Args:
        | returns (pd.DataFrame): Historical price data a ticker.
        | strategy_name (str): Name of the strategy.
        | param_grid: Grid of parameters for the strategy.
        | train_lookback (int): Size of training window (in rows).
        | train_step (int): Step size for re-optimization.
        | seed: Seed for random selection.
    """
    perm_wf = get_permutation(returns.copy(), train_lookback, seed)
    perm_log_returns = np.log(perm_wf["Close"]).diff().shift(-1)
    perm_wf_signal = walkforward_test(
        perm_wf, strategy_name, param_grid, train_lookback, train_step
    )
    perm_pf, _ = evaluate_strategy(perm_wf_signal, perm_log_returns)
    return perm_pf


def walkforward_permutation_test_parallelised(
    returns: pd.DataFrame,
    strategy_name: str,
    param_grid,
    n_permutations: int = 100,
    seed=None,
    train_lookback: int = 252 * 4,
    train_step: int = 21,
    n_jobs: int = -1,  # Use all available CPU cores
):
    """
    Fourth step in strategy testing - run a walkforward permutation test.
    Uses out-of-sample data for evaluating a strategy, optimising it based on a time window and
    then using this to perform trades in the next time window, using permutations.
    This is quite slow, hence the 100 default permutations used.
    Args:
        | returns (pd.DataFrame): Historical price data a ticker.
        | strategy_name (str): Name of the strategy.
        | param_grid: Grid of parameters for the strategy.
        | n_permutations (int): Number of permutations to use (recommended is 100).
        | seed: Seed for random selection.
        | train_lookback (int): Size of training window (in rows).
        | train_step (int): Step size for re-optimization.
        | n_jobs (int): Used for parallelising execution.
    """
    log_returns = np.log(returns["Close"]).diff().shift(-1)
    wf_signal = walkforward_test(
        returns, strategy_name, param_grid, train_lookback, train_step
    )
    real_pf, real_sharpe = evaluate_strategy(wf_signal, log_returns)
    print(f"Walkforward PF = {real_pf}")
    np.random.seed(seed)
    seeds = np.random.randint(0, 1e6, size=n_permutations)
    perm_pfs = Parallel(n_jobs=n_jobs)(
        delayed(single_wf_permutation_test)(
            returns, strategy_name, param_grid, train_lookback, train_step, s
        )
        for s in seeds
    )
    perm_better_count = sum([pf >= real_pf for pf in perm_pfs])
    wf_mcpt_pval = perm_better_count / n_permutations
    print(f"Walkforward MCPT P-Value = {wf_mcpt_pval:.4f}")
    # Plot
    plt.style.use("dark_background")
    pd.Series(perm_pfs).hist(color="blue", label="Permuted PFs")
    plt.axvline(real_pf, color="red", linestyle="--", label="Real PF")
    plt.title(f"Walkforward MCPT\nP-Value: {wf_mcpt_pval:.4f}")
    plt.xlabel("Profit Factor")
    plt.legend()
    plt.show()
    return wf_mcpt_pval


def walkforward_permutation_test(
    returns: pd.DataFrame,
    strategy_name: str,
    param_grid,
    n_permutations: int = 10,
    seed=None,
    train_lookback: int = 252 * 4,  # 4 years of daily data
    train_step: int = 21,  # Step forward by ~1 month
):
    """
    Fourth step in strategy testing - run a walkforward permutation test.
    Uses out-of-sample data for evaluating a strategy, optimising it based on a time window and
    then using this to perform trades in the next time window. It uses permutations to
    Args:
        | returns (pd.DataFrame): Historical price data a ticker.
        | strategy_name (str): Name of the strategy.
        | param_grid: Grid of parameters for the strategy.
        | n_permutations (int): Number of permutations to use (recommended is 1000).
        | start_index (int): Index to start permutation.
        | seed: Seed for random selection.
        | train_lookback (int): Size of training window (in rows).
        | train_step (int): Step size for re-optimization.
    """
    log_returns = np.log(returns["Close"]).diff().shift(-1)
    wf_signal = walkforward_test(
        returns, strategy_name, param_grid, train_lookback, train_step
    )
    real_wf_pf, real_wf_sharpe = evaluate_strategy(wf_signal, log_returns)
    print(f"Walkforward PF = {real_wf_pf}.")
    perm_better_count = 1
    permuted_profit_factors = []
    for _ in range(1, n_permutations):
        perm_wf = get_permutation(returns, train_lookback, seed)
        perm_log_returns = np.log(perm_wf["Close"]).diff().shift(-1)
        perm_wf_signal = walkforward_test(
            perm_wf, strategy_name, param_grid, train_lookback, train_step
        )
        perm_wf_pf, perm_wf_sharpe = evaluate_strategy(perm_wf_signal, perm_log_returns)
        if perm_wf_pf >= real_wf_pf:
            perm_better_count += 1
        permuted_profit_factors.append(perm_wf_pf)
    wf_mcpt_pval = perm_better_count / n_permutations
    print(f"Walkforward MCPT P-Value = {wf_mcpt_pval}")
    plt.style.use("dark_background")
    pd.Series(permuted_profit_factors).hist(color="blue", label="Permutations")
    plt.axvline(real_wf_pf, color="red", label="Real")
    plt.xlabel("Profit Factor")
    plt.title(f"Walkforward MCPT. P-Value: {wf_mcpt_pval}")
    plt.grid(False)
    plt.legend()
    plt.show()


# ---------------------------- PLOT LOG RETURNS ---------------------------- #


def plot_strategy_cumulative_log_returns(
    returns: pd.DataFrame, signal: pd.Series, strategy_name
):
    """
    This function plots the log cumulative returns of a strategy on some price time series data.
    Args:
        | returns (pd.DataFrame): Historical price data a ticker.
        | signal (pd.Series): Vector describing positions taken by the strategy through time.
        | strategy_name (str): The name of the strategy.
    """
    log_returns = np.log(returns["Close"]).diff().shift(-1)
    log_returns["strategy_returns"] = log_returns * signal
    plt.figure()
    log_returns["strategy_returns"].cumsum().plot(color="red")
    plt.title("In-Sample " + strategy_name)
    plt.ylabel("Cumulative Log Return")
    plt.show(block=False)


# ---------------------------- MAIN ---------------------------- #

# Run in-sample initial test - optimise strategies
"""
if __name__ == "__main__":
    ticker = "GBPUSD=X"
    start_date = "2016-01-01"
    end_date = "2021-01-01"
    returns_df = yf.download(ticker, start=start_date, end=end_date)
    returns_df.columns = returns_df.columns.get_level_values(0)
    param_grid_donchian = [{"lookback": x} for x in range(5, 250)]
    param_grid_ma = [
        {"short": s, "long": l} for s in range(5, 90) for l in range(90, 365) if s < l
    ]
    param_grid_mr = [{"lookback": x} for x in range(10, 365)]
    for strat, grid in zip(
        ["donchian", "ma_cross", "mean_revert"],
        [param_grid_donchian, param_grid_ma, param_grid_mr],
    ):
        params, pf, signal = run_backtest(returns_df, strat, grid)
        print(f"Best {strat} strategy:")
        print(f"  Parameters: {params}")
        print(f"  Profit Factor: {pf:.2f}\n")
        plt.style.use("dark_background")
        plot_strategy_cumulative_log_returns(returns_df, signal, strat)
    plt.show()
"""


# Run in-sample permutation test setup - verify permutation
"""
if __name__ == "__main__":
    ticker = "GBPUSD=X"
    start_date = "2016-01-01"
    end_date = "2021-01-01"
    returns_df = yf.download(ticker, start=start_date, end=end_date)
    returns_df.columns = returns_df.columns.get_level_values(0)
    gbpusd_perm = get_permutation(returns_df)
    gbpusd_real_returns = np.log(returns_df["Close"]).diff()
    gbpusd_perm_returns = np.log(gbpusd_perm["Close"]).diff()
    print(
        f"GBP/USD Mean. REAL: {gbpusd_real_returns.mean():14.6f} PERM: {gbpusd_perm_returns.mean():14.6f}"
    )
    print(
        f"GBP/USD Stdd. REAL: {gbpusd_real_returns.std():14.6f} PERM: {gbpusd_perm_returns.std():14.6f}"
    )
    print(
        f"GBP/USD Skew. REAL: {gbpusd_real_returns.skew():14.6f} PERM: {gbpusd_perm_returns.skew():14.6f}"
    )
    print(
        f"GBP/USD Kurt. REAL: {gbpusd_real_returns.kurt():14.6f} PERM: {gbpusd_perm_returns.kurt():14.6f}"
    )
    plt.style.use("dark_background")
    np.log(returns_df["Close"]).diff().cumsum().plot(
        color="orange", label="Real GBP/USD"
    )
    np.log(gbpusd_perm["Close"]).diff().cumsum().plot(
        color="purple", label="Permuted GBP/USD"
    )
    plt.ylabel("Cumulative Log Return")
    plt.title("Real and Permuted GBP/USD")
    plt.legend()
    plt.show()
"""


# Run in-Sample permutation test
"""
if __name__ == "__main__":
    ticker = "GBPUSD=X"
    start_date = "2016-01-01"
    end_date = "2021-01-01"
    returns_df = yf.download(ticker, start=start_date, end=end_date)
    returns_df.columns = returns_df.columns.get_level_values(0)
    permutations = 1000
    strategy_name = "donchian"
    param_grid_donchian = [{"lookback": x} for x in range(5, 250)]
    plt.style.use("dark_background")
    permutation_test(returns_df, strategy_name, param_grid_donchian, permutations)
"""


# Run walkforward test

"""if __name__ == "__main__":
    ticker = "GBPUSD=X"
    start_date = "2016-01-01"
    end_date = "2021-01-01"
    returns_df = yf.download(ticker, start=start_date, end=end_date)
    returns_df.columns = returns_df.columns.get_level_values(0)
    plt.style.use("dark_background")
    param_grid_donchian = [{"lookback": x} for x in range(5, 250)]
    param_grid_ma = [
        {"short": s, "long": l} for s in range(5, 90) for l in range(90, 365) if s < l
    ]
    param_grid_mr = [{"lookback": x} for x in range(10, 365)]
    for strat, grid in zip(
        ["donchian", "ma_cross", "mean_revert"],
        [param_grid_donchian, param_grid_ma, param_grid_mr],
    ):
        print(f"Running walkforward test for {strat} strategy...")
        wf_signal = walkforward_test(returns_df, strat, grid)
        pf, sharpe = evaluate_strategy(
            wf_signal, np.log(returns_df["Close"]).diff().shift(-1)
        )
        print(f"Walkforward {strat} strategy:")
        print(f"  Profit Factor: {pf:.2f}")
        print(f"  Sharpe Ratio: {sharpe:.2f}\n")
        plot_strategy_cumulative_log_returns(returns_df, wf_signal, f"{strat} (WF)")
    plt.show()"""


# Run walkforward permutation test

if __name__ == "__main__":
    ticker = "GBPUSD=X"
    start_date = "2016-01-01"
    end_date = "2021-01-01"
    returns_df = yf.download(ticker, start=start_date, end=end_date)
    returns_df.columns = returns_df.columns.get_level_values(0)
    plt.style.use("dark_background")
    param_grid_donchian = [{"lookback": x} for x in range(5, 250)]
    param_grid_mr = [{"lookback": x} for x in range(10, 365)]
    for strat, grid in zip(
        ["donchian", "mean_revert"],
        [param_grid_donchian, param_grid_mr],
    ):
        print(f"Running walkforward test for {strat} strategy...")
        wf_signal = walkforward_permutation_test_parallelised(returns_df, strat, grid)
        """pf, sharpe = evaluate_strategy(
            wf_signal, np.log(returns_df["Close"]).diff().shift(-1)
        )
        print(f"Walkforward {strat} strategy:")
        print(f"  Profit Factor: {pf:.2f}")
        print(f"  Sharpe Ratio: {sharpe:.2f}\n")
        plot_strategy_cumulative_log_returns(returns_df, wf_signal, f"{strat} (WF)")"""
    plt.show()
