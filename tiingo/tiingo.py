import torch
import numpy as np
import pandas as pd
from typing import Tuple
from dataset.ts_dataset import TimeSeriesRegressionDataset
from tiingo import TiingoClient

TIINGO_API_KEY = "63a212473489b88a0406ce83cad2a801ef188bf0"
TIINGO_BASE_URL = "https://api.tiingo.com/tiingo/daily/"
TIINGO_CONFIG = {"session": True, "api_key": TIINGO_API_KEY}


def get_adj_close_px(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    tiingo = TiingoClient(TIINGO_CONFIG)
    return tiingo.get_dataframe(
        tickers, metric_name="adjClose", startDate=start_date, endDate=end_date
    )


def get_daily_returns(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    data = get_adj_close_px(tickers, start_date, end_date)
    data = data.pct_change()
    return data


def prepare_cross_sectional_data(
    df_returns: pd.DataFrame,
    window_size: int,
    train_split: float = 0.7,
    val_split: float = 0.15,
) -> Tuple[
    TimeSeriesRegressionDataset,
    TimeSeriesRegressionDataset,
    TimeSeriesRegressionDataset,
]:
    """
    Fetches data and creates sliding windows suitable for the CrossSectionalAlphaTransformer.

    The resulting tensors will have shapes:
    X: (Num_Samples, N_Stocks, Window_Size)
    y: (Num_Samples, N_Stocks, 1)
    """
    # Simple data cleaning: fill NaNs with 0 (or use forward fill in production)
    df_returns = df_returns.fillna(0.0)

    raw_data = df_returns.values  # Shape: (Total_Days, N_Stocks)

    X_list = []
    y_list = []

    # 2. Create Sliding Windows
    # We need 'window_size' days for features, and the next day for target
    num_samples = len(raw_data) - window_size

    for i in range(num_samples):
        # Input: Window of 'window_size' days for ALL stocks
        # Shape from slice: (Window_Size, N_Stocks)
        # We transpose to get: (N_Stocks, Window_Size) which matches model input (N, T)
        window = raw_data[i : i + window_size].T

        # Target: The return on the very next day
        # Shape from slice: (N_Stocks,)
        # We reshape to (N_Stocks, 1) to match output dimension
        target = raw_data[i + window_size].reshape(-1, 1)

        X_list.append(window)
        y_list.append(target)

    # Stack into single tensors
    X_all = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_all = torch.tensor(np.array(y_list), dtype=torch.float32)

    # 3. Chronological Split (Train -> Val -> Test)
    total_samples = len(X_all)
    train_idx = int(total_samples * train_split)
    val_idx = int(total_samples * (train_split + val_split))

    X_train, y_train = X_all[:train_idx], y_all[:train_idx]
    X_val, y_val = X_all[train_idx:val_idx], y_all[train_idx:val_idx]
    X_test, y_test = X_all[val_idx:], y_all[val_idx:]

    # 4. Create Datasets
    train_ds = TimeSeriesRegressionDataset(X_train, y_train)
    val_ds = TimeSeriesRegressionDataset(X_val, y_val)
    test_ds = TimeSeriesRegressionDataset(X_test, y_test)

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    df_ret = get_daily_returns(["AAPL"], start_date="2012-01-01", end_date="2025-11-20")
    print(df_ret.head())
    print(df_ret.tail())


# tiingo.get_dataframe(["AAPL", "MSFT", "VST", "GOOG", "RACE", "JPM", "GS", "NVDA", "CRM", "LLY", "TSLA"], metric_name="adjClose", startDate="2012-01-01", endDate="2025-11-20").pct_change().cumsum().plot()
