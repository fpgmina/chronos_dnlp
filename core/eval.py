import torch
import numpy as np
from tqdm import tqdm
from utils import compute_metrics


def chronos_one_step_forecast(
    pipeline, window_df, device, quantile_idx=4  # shape (T, N)  # default = median
):
    """
    Returns:
        y_hat: (N,) predicted next-day returns
    """

    # (T, N) → (N, T) → (1, N, T)
    context = (
        torch.tensor(window_df.values.T, dtype=torch.float32).unsqueeze(0).to(device)
    )

    forecast = pipeline.predict(
        context,
        prediction_length=1,
    )

    # (N, 9, 1) → (N,)
    y_hat = forecast[0][:, quantile_idx, 0].cpu().numpy()

    return y_hat


def run_chronos_sliding_backtest(
    pipeline,
    df_returns,  # full TxN returns DataFrame
    device,
    context_length=200,
    start_idx=None,
    quantile_idx=4,
):
    """
    Walk-forward evaluation of Chronos.

    Returns:
        pred_matrix: (T_eval, N)
        true_matrix: (T_eval, N)
        daily_metrics: list of metric dicts
    """

    df = df_returns.dropna().copy()
    T, N = df.shape

    if start_idx is None:
        start_idx = context_length

    preds = []
    trues = []
    daily_metrics = []
    timestamps = []

    for t in tqdm(range(start_idx, T - 1)):

        window = df.iloc[t - context_length : t]  # (T, N)
        y_true = df.iloc[t].values  # (N,)

        y_pred = chronos_one_step_forecast(
            pipeline=pipeline,
            window_df=window,
            device=device,
            quantile_idx=quantile_idx,
        )

        preds.append(y_pred)
        trues.append(y_true)
        timestamps.append(df.index[t])

        metrics = compute_metrics(y_true, y_pred)
        daily_metrics.append(metrics)

    pred_matrix = np.vstack(preds)
    true_matrix = np.vstack(trues)

    return {
        "preds": pred_matrix,
        "trues": true_matrix,
        "dates": np.array(timestamps),
        "daily_metrics": daily_metrics,
    }


def summarize_backtest_results(results):

    daily_metrics = results["daily_metrics"]

    agg = {
        "mean_corr": np.nanmean([m["corr"] for m in daily_metrics]),
        "mean_r2": np.nanmean([m["r2"] for m in daily_metrics]),
        "mean_mse": np.nanmean([m["mse"] for m in daily_metrics]),
        "mean_mae": np.nanmean([m["mae"] for m in daily_metrics]),
    }

    return agg
