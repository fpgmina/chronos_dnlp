"""
Data preparation utilities for Chronos-2 multivariate fine-tuning.
"""

from typing import Iterable, Mapping

import numpy as np
import pandas as pd


GICS_LEVEL_1 = {
    "Information Technology": [
        "AAPL",
        "ADBE",
        "ADI",
        "AMD",
        "AMAT",
        "CSCO",
        "FIS",
        "IBM",
        "INTC",
        "INTU",
        "LRCX",
        "MSFT",
        "MU",
        "NVDA",
        "ORCL",
        "QCOM",
        "TXN",
    ],
    "Health Care": [
        "ABBV",
        "ABT",
        "AMGN",
        "BDX",
        "BMY",
        "CVS",
        "DHR",
        "ELV",
        "GILD",
        "HUM",
        "ISRG",
        "JNJ",
        "LLY",
        "MRK",
        "PFE",
        "REGN",
        "SYK",
        "TMO",
        "VRTX",
        "ZTS",
    ],
    "Financials": [
        "AIG",
        "AXP",
        "BAC",
        "BLK",
        "C",
        "CB",
        "CI",
        "COF",
        "GS",
        "JPM",
        "MET",
        "MS",
        "PNC",
        "PGR",
        "SCHW",
        "USB",
        "WFC",
    ],
    "Consumer Discretionary": [
        "AMZN",
        "BKNG",
        "HD",
        "LOW",
        "MCD",
        "META",
        "NFLX",
        "NKE",
        "SBUX",
        "TGT",
        "TJX",
    ],
    "Consumer Staples": [
        "CL",
        "COST",
        "EL",
        "KMB",
        "KO",
        "MDLZ",
        "MO",
        "PEP",
        "PG",
        "PM",
        "WMT",
    ],
    "Industrials": [
        "BA",
        "CAT",
        "CSX",
        "DE",
        "EMR",
        "ETN",
        "FDX",
        "GD",
        "GE",
        "HON",
        "ITW",
        "LMT",
        "MMM",
        "NSC",
        "RTX",
        "UNP",
        "UPS",
        "WM",
    ],
    "Energy": ["COP", "CVX", "EOG", "OXY", "SLB", "XOM"],
    "Communication Services": ["CMCSA", "CRM", "GOOG", "GOOGL", "T", "TMUS", "VZ"],
    "Materials": ["APD", "LIN", "SHW"],
    "Real Estate": ["AMT", "SPG"],
    "Utilities": [
        # nessuna delle aziende fornite rientra qui
    ],
}


def create_multivariate_windows(
    df: pd.DataFrame,
    context_length: int,
    prediction_length: int,
    stride: int = 50,
    group_id: int | None = None,
) -> list[Mapping[str, np.ndarray]]:
    """
    Create multiple multivariate training windows from a wide DataFrame.

    Each window contains all stocks' data for a contiguous time period,
    enabling Chronos-2 to learn cross-stock patterns via group attention.

    Args:
        df: Wide DataFrame with shape (T, N) where rows are dates and columns are stocks.
            Should be cleaned (no NaNs) before calling this function.
        context_length: Number of historical timesteps the model sees as input.
        prediction_length: Number of future timesteps the model predicts.
        stride: Step size between consecutive windows. Smaller stride = more windows
            but more overlap. Default 50.
        group_id: Optional group identifier to attach to each sample. When provided,
            Chronos-2 can use this to separate groups for multivariate attention.

    Returns:
        List of dicts, each with key "target" mapping to array of shape
        (N, context_length + prediction_length). Each dict is one training sample
        where all N stocks are jointly modeled. If group_id is provided, each sample
        also includes a "group_id" field.

    Example:
        >>> df = pd.DataFrame(np.random.randn(1000, 50))  # 1000 days, 50 stocks
        >>> windows = create_multivariate_windows(df, context_length=200, prediction_length=1)
        >>> len(windows)  # ~16 windows with stride=50
        16
        >>> windows[0]["target"].shape
        (50, 201)
    """
    data = df.values.T.astype(np.float32)  # (N, T)
    N, T = data.shape
    window_len = context_length + prediction_length

    inputs: list[Mapping[str, np.ndarray]] = []
    for start in range(0, T - window_len + 1, stride):
        window = data[:, start : start + window_len]  # (N, window_len)
        sample: dict[str, np.ndarray] = {"target": window}
        if group_id is not None:
            sample["group_id"] = np.int64(group_id)
        inputs.append(sample)

    return inputs


def create_grouped_multivariate_windows(
    df: pd.DataFrame,
    groups: Mapping[str, Iterable[str]] | None,
    context_length: int,
    prediction_length: int,
    stride: int = 50,
    group_id_offset: int = 0,
) -> list[Mapping[str, np.ndarray]]:
    """
    Create multivariate training windows per group with distinct group IDs.

    Each group is trained on its own subset of tickers, and every sample is
    tagged with a unique group_id to enable Chronos-2 group-aware attention.

    Args:
        df: Wide DataFrame with shape (T, N) where rows are dates and columns are stocks.
            Should be cleaned (no NaNs) before calling this function.
        groups: Mapping of group name -> iterable of ticker symbols. When None, uses GICS_LEVEL_1.
        context_length: Number of historical timesteps the model sees as input.
        prediction_length: Number of future timesteps the model predicts.
        stride: Step size between consecutive windows. Default 50.
        group_id_offset: Starting integer for group IDs. Default 0.

    Returns:
        List of dicts with "target" arrays and "group_id" values.
    """
    # MODIFIED FUNCTION
    # MODIFIED LINES
    groups = GICS_LEVEL_1 if groups is None else groups
    inputs: list[Mapping[str, np.ndarray]] = []
    for group_idx, (_, tickers) in enumerate(groups.items()):
        available = [ticker for ticker in tickers if ticker in df.columns]
        if not available:
            continue
        group_df = df[available]
        inputs.extend(
            create_multivariate_windows(
                group_df,
                context_length=context_length,
                prediction_length=prediction_length,
                stride=stride,
                group_id=group_id_offset + group_idx,
            )
        )
    return inputs


def train_val_split(
    inputs: list[Mapping[str, np.ndarray]],
    val_ratio: float = 0.1,
) -> tuple[list[Mapping[str, np.ndarray]], list[Mapping[str, np.ndarray]]]:
    """
    Split training inputs into train and validation sets.

    Uses the last portion of inputs for validation to preserve temporal ordering
    (earlier windows for training, later windows for validation).

    Args:
        inputs: List of training samples from create_multivariate_windows.
        val_ratio: Fraction of samples to use for validation. Default 0.1 (10%).

    Returns:
        Tuple of (train_inputs, val_inputs).

    Example:
        >>> inputs = [{"target": np.random.randn(50, 201)} for _ in range(100)]
        >>> train, val = train_val_split(inputs, val_ratio=0.2)
        >>> len(train), len(val)
        (80, 20)
    """
    split_idx = int(len(inputs) * (1 - val_ratio))
    return inputs[:split_idx], inputs[split_idx:]


def prepare_data_for_chronos(
    df: pd.DataFrame,
    test_size: int = 1200,
    ffill_limit: int = 5,
    bfill_limit: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare raw stock returns data for Chronos-2 training.

    Splits into train/test, handles missing values, and ensures consistent columns.

    Args:
        df: Wide DataFrame with shape (T, N) where rows are dates and columns are stocks.
        test_size: Number of rows to reserve for testing. Default 1200.
        ffill_limit: Maximum consecutive NaNs to forward fill. Default 5.
        bfill_limit: Maximum consecutive NaNs to backward fill. Default 5.

    Returns:
        Tuple of (df_train_clean, df_test_clean) with no NaN values.

    Example:
        >>> df = get_daily_returns_data_cached()
        >>> df_train, df_test = prepare_data_for_chronos(df, test_size=1200)
    """
    # Split: train = all except last test_size rows
    df_train = df.iloc[:-test_size]
    df_test = df.iloc[-test_size:]

    # Clean data: forward fill, backward fill, then drop any remaining NaN columns
    df_train_clean = (
        df_train.ffill(limit=ffill_limit).bfill(limit=bfill_limit).dropna(axis=1)
    )

    # Ensure test has same columns and clean it
    df_test_clean = (
        df_test[df_train_clean.columns]
        .ffill(limit=ffill_limit)
        .bfill(limit=bfill_limit)
        .dropna()
    )

    return df_train_clean, df_test_clean
