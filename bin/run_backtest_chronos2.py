import torch
from chronos import Chronos2Pipeline
from core.eval import run_chronos_sliding_backtest, summarize_backtest_results
from tiingo_data.download_data import get_daily_returns_data_cached
from utils import get_device
import warnings

warnings.filterwarnings(
    "ignore",
    message="'pin_memory' argument is set as true but not supported on MPS now",
)


def run():
    df = get_daily_returns_data_cached().iloc[-1200:]

    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=get_device(),
        torch_dtype=torch.float32,
    )
    results = run_chronos_sliding_backtest(
        pipeline=pipeline,
        df_returns=df,
        device=get_device(),
        context_length=200,
    )

    summary = summarize_backtest_results(results)

    print("==== GLOBAL SUMMARY ====")
    print(summary)


if __name__ == "__main__":
    run()


# python -m bin.run_backtest_chronos2
