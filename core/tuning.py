"""
Hyperparameter tuning utilities for Chronos-2 fine-tuning.

Ottimizzazioni incluse:
- Riduzione frammentazione VRAM (PYTORCH_CUDA_ALLOC_CONF) impostata QUI (best-effort).
- Forzatura GPU quando disponibile (device_map su cuda:0).
- Mixed precision (bf16 se supportato, altrimenti fp16) per velocità + meno VRAM.
- Cleanup robusto con try/finally, synchronize, empty_cache, gc.collect.
- Eval in torch.inference_mode() per evitare grafi e ridurre memoria.
- Log debug opzionale: esporta CHRONOS_TUNING_DEBUG=1.
"""

from __future__ import annotations

# =========================
# "Parte prima" incorporata
# =========================
import os

# Best-effort: se non è già impostata, riduce frammentazione e rallentamenti progressivi.
# NOTA: idealmente andrebbe settata PRIMA di importare torch in tutto il processo.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128"
)

# Debug opzionale (export CHRONOS_TUNING_DEBUG=1)
_DEBUG = os.environ.get("CHRONOS_TUNING_DEBUG", "0") == "1"

# =========================
# Imports
# =========================
import gc
import itertools
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import optuna
import torch
from chronos import Chronos2Pipeline

from core.data import (
    GICS_LEVEL_1,
    create_multivariate_windows,
    prepare_data_for_chronos,
    train_val_split,
)
from core.eval import evaluate_model_on_test


# Performance tweaks CUDA (sicuri e spesso utili)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


@dataclass(frozen=True)
class TuningResult:
    group_name: str
    config: dict[str, int | float]
    mean_quantile_loss: float
    mean_mse: float
    mean_mae: float


def _dbg(msg: str) -> None:
    if _DEBUG:
        print(f"[CHRONOS_TUNING_DEBUG] {msg}")


def _clear_memory() -> None:
    """
    Cleanup aggressivo: aiuta a evitare rallentamenti dopo molti trial.
    """
    gc.collect()

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def _iter_param_grid(search_space: Mapping[str, Sequence[int | float]]):
    keys = list(search_space.keys())
    values = [search_space[key] for key in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def _suggest_params(
    trial: optuna.Trial,
    search_space: Mapping[str, Mapping[str, int | float | str]],
) -> dict[str, int | float]:
    params: dict[str, int | float] = {}
    for name, spec in search_space.items():
        kind = spec["type"]
        if kind == "int":
            params[name] = trial.suggest_int(
                name, int(spec["low"]), int(spec["high"]), step=int(spec.get("step", 1))
            )
        elif kind == "float":
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
        else:
            raise ValueError(f"Unsupported param type: {kind} for {name}")
    return params


def _cleanup_artifacts(output_dir: str | Path | None) -> None:
    if output_dir is None:
        return
    path = Path(output_dir)
    if path.exists():
        shutil.rmtree(path)


def _device_map_and_dtype() -> tuple[object, torch.dtype, str]:
    """
    Forza device e dtype per essere CERTI di usare GPU quando disponibile.
    - CUDA: device_map={"":0} => tutto su GPU 0 (evita fallback a CPU).
    - dtype: bf16 se supportato, altrimenti fp16.
    - CPU: fp32.
    """
    if torch.cuda.is_available():
        device_map = {"": 0}  # forza tutto su cuda:0
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return device_map, dtype, "cuda:0"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS: lasciamo semplice
        return "mps", torch.float16, "mps"

    return "cpu", torch.float32, "cpu"


def _assert_pipeline_on_expected_device(pipeline, expected_device_label: str) -> None:
    """
    Verifica best-effort che il modello stia sul device previsto (soprattutto CUDA).
    """
    # Se non c'è CUDA, non forziamo
    if expected_device_label.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("expected_device_label=cuda ma torch.cuda non è disponibile.")

    model = getattr(pipeline, "model", None) or getattr(pipeline, "_model", None)
    if model is None:
        _dbg("pipeline.model non trovato: impossibile verificare device.")
        return

    try:
        p = next(model.parameters())
    except StopIteration:
        _dbg("Modello senza parametri: impossibile verificare device.")
        return

    if expected_device_label.startswith("cuda"):
        if p.device.type != "cuda":
            raise RuntimeError(
                f"Il modello NON è su GPU: device={p.device}. Atteso ~ {expected_device_label}"
            )
    else:
        # best-effort per mps/cpu
        if p.device.type == "cuda" and not expected_device_label.startswith("cuda"):
            _dbg(f"Modello su CUDA inatteso: {p.device}")


def _log_cuda_memory(prefix: str) -> None:
    if not torch.cuda.is_available() or not _DEBUG:
        return
    alloc = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    peak = torch.cuda.max_memory_allocated() / (1024**2)
    _dbg(f"{prefix} | allocated={alloc:.1f}MB reserved={reserved:.1f}MB peak={peak:.1f}MB")


def _train_and_eval(
    train_inputs,
    val_inputs,
    val_df,
    config: Mapping[str, int | float],
    output_dir: str | Path | None,
) -> TuningResult:
    device_map, dtype, device_label = _device_map_and_dtype()

    # Pulizia preventiva
    _clear_memory()
    if torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device_map,
        dtype=dtype,
    )

    _assert_pipeline_on_expected_device(pipeline, device_label)
    _log_cuda_memory("after from_pretrained")

    finetuned = None
    eval_results = None

    try:
        finetuned = pipeline.fit(
            inputs=train_inputs,
            validation_inputs=val_inputs,
            prediction_length=1,
            context_length=int(config["context_length"]),
            min_past=int(config["context_length"]),
            num_steps=int(config["num_steps"]),
            batch_size=int(config["batch_size"]),
            learning_rate=float(config["learning_rate"]),
            output_dir=str(output_dir) if output_dir else None,
        )

        _log_cuda_memory("after fit")

        # Eval senza gradienti/grafo
        with torch.inference_mode():
            eval_results = evaluate_model_on_test(
                pipeline=finetuned,
                df_test=val_df,
                context_length=int(config["context_length"]),
                n_samples=int(config.get("n_eval_samples", 100)),
            )

        _log_cuda_memory("after eval")

        result = TuningResult(
            group_name=str(config.get("group_name", "global")),
            config=dict(config),
            mean_quantile_loss=float(eval_results["mean_quantile_loss"]),
            mean_mse=float(eval_results["mean_mse"]),
            mean_mae=float(eval_results["mean_mae"]),
        )
        return result

    finally:
        # Cleanup SEMPRE, anche se un trial fallisce
        try:
            del eval_results
        except Exception:
            pass
        try:
            del finetuned
        except Exception:
            pass
        try:
            del pipeline
        except Exception:
            pass

        _clear_memory()
        _log_cuda_memory("after cleanup")


def tune_hyperparams_for_dataframe(
    df,
    search_space: Mapping[str, Sequence[int | float]],
    group_name: str = "global",
    test_size: int = 1200,
    val_ratio: float = 0.1,
    output_dir: str | Path | None = "outputs/chronos2_tuning",
    cleanup_output_dir: bool = True,
) -> tuple[TuningResult, list[TuningResult]]:
    """
    Tune Chronos-2 hyperparameters on a single wide DataFrame (grid search).

    Args:
        df: Wide DataFrame (T, N) of returns.
        search_space: Mapping of hyperparameter -> list of values. Must include
            context_length, prediction_length, learning_rate, num_steps, batch_size.
            Optionally include stride and n_eval_samples.
        group_name: Label for the dataset.
        test_size: Number of rows to hold out for evaluation.
        val_ratio: Fraction of training windows reserved for validation.
        output_dir: Directory for model artifacts.
        cleanup_output_dir: Whether to delete artifacts after each trial.
    """
    train_df, val_df = prepare_data_for_chronos(df, test_size=test_size)

    results: list[TuningResult] = []
    best_result: TuningResult | None = None

    for trial_idx, config in enumerate(_iter_param_grid(search_space), start=1):
        config = dict(config)
        config["group_name"] = group_name
        config.setdefault("stride", 50)
        config.setdefault("n_eval_samples", 100)

        train_inputs = None
        val_inputs = None
        run_output_dir = None

        try:
            train_inputs = create_multivariate_windows(
                train_df,
                context_length=int(config["context_length"]),
                prediction_length=1,
                stride=int(config["stride"]),
            )
            train_inputs, val_inputs = train_val_split(train_inputs, val_ratio=val_ratio)

            if output_dir is not None:
                run_output_dir = Path(output_dir) / f"{group_name}_trial_{trial_idx}"

            result = _train_and_eval(
                train_inputs=train_inputs,
                val_inputs=val_inputs,
                val_df=val_df,
                config=config,
                output_dir=run_output_dir,
            )
            results.append(result)

            if best_result is None or result.mean_quantile_loss < best_result.mean_quantile_loss:
                best_result = result

        finally:
            # libera window data ASAP
            try:
                del train_inputs
            except Exception:
                pass
            try:
                del val_inputs
            except Exception:
                pass

            _clear_memory()
            if cleanup_output_dir:
                _cleanup_artifacts(run_output_dir)

    if best_result is None:
        raise ValueError("No tuning results generated. Check the search space.")

    return best_result, results


def tune_hyperparams_by_group(
    df,
    search_space: Mapping[str, Sequence[int | float]],
    groups: Mapping[str, Iterable[str]] | None = None,
    test_size: int = 1200,
    val_ratio: float = 0.1,
    output_dir: str | Path | None = "outputs/chronos2_tuning",
    cleanup_output_dir: bool = True,
) -> dict[str, tuple[TuningResult, list[TuningResult]]]:
    """
    Tune hyperparameters for each group (e.g., GICS sectors) and the global set.

    Returns:
        Dict mapping group name to (best_result, all_results).
    """
    groups = GICS_LEVEL_1 if groups is None else groups
    results: dict[str, tuple[TuningResult, list[TuningResult]]] = {}

    global_best, global_results = tune_hyperparams_for_dataframe(
        df=df,
        search_space=search_space,
        group_name="global",
        test_size=test_size,
        val_ratio=val_ratio,
        output_dir=output_dir,
        cleanup_output_dir=cleanup_output_dir,
    )
    results["global"] = (global_best, global_results)

    for group_name, tickers in groups.items():
        available = [ticker for ticker in tickers if ticker in df.columns]
        if not available:
            continue
        group_df = df[available]

        best, all_results = tune_hyperparams_for_dataframe(
            df=group_df,
            search_space=search_space,
            group_name=group_name,
            test_size=test_size,
            val_ratio=val_ratio,
            output_dir=output_dir,
            cleanup_output_dir=cleanup_output_dir,
        )
        results[group_name] = (best, all_results)
        _clear_memory()

    return results


def tune_hyperparams_for_dataframe_optuna(
    df,
    search_space: Mapping[str, Mapping[str, int | float | str]],
    group_name: str = "global",
    test_size: int = 1200,
    val_ratio: float = 0.1,
    output_dir: str | Path | None = "outputs/chronos2_tuning",
    cleanup_output_dir: bool = True,
    n_trials: int = 20,
    sampler: optuna.samplers.BaseSampler | None = None,
) -> tuple[TuningResult, list[TuningResult]]:
    """
    Tune Chronos-2 hyperparameters with Optuna on a single wide DataFrame.
    """
    train_df, val_df = prepare_data_for_chronos(df, test_size=test_size)

    results: list[TuningResult] = []
    best_result: TuningResult | None = None

    def objective(trial: optuna.Trial) -> float:
        nonlocal best_result

        _clear_memory()
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

        config = _suggest_params(trial, search_space)
        config["group_name"] = group_name
        config.setdefault("stride", 50)
        config.setdefault("n_eval_samples", 100)

        train_inputs = None
        val_inputs = None
        run_output_dir = None

        try:
            train_inputs = create_multivariate_windows(
                train_df,
                context_length=int(config["context_length"]),
                prediction_length=1,
                stride=int(config["stride"]),
            )
            train_inputs, val_inputs = train_val_split(train_inputs, val_ratio=val_ratio)

            if output_dir is not None:
                run_output_dir = Path(output_dir) / f"{group_name}_trial_{trial.number + 1}"

            result = _train_and_eval(
                train_inputs=train_inputs,
                val_inputs=val_inputs,
                val_df=val_df,
                config=config,
                output_dir=run_output_dir,
            )
            results.append(result)

            if best_result is None or result.mean_quantile_loss < best_result.mean_quantile_loss:
                best_result = result

            return result.mean_quantile_loss

        finally:
            # libera window data ASAP
            try:
                del train_inputs
            except Exception:
                pass
            try:
                del val_inputs
            except Exception:
                pass

            _clear_memory()
            if cleanup_output_dir:
                _cleanup_artifacts(run_output_dir)

    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    if best_result is None:
        raise ValueError("No tuning results generated. Check the search space.")

    return best_result, results


def tune_hyperparams_by_group_optuna(
    df,
    search_space: Mapping[str, Mapping[str, int | float | str]],
    groups: Mapping[str, Iterable[str]] | None = None,
    test_size: int = 1200,
    val_ratio: float = 0.1,
    output_dir: str | Path | None = "outputs/chronos2_tuning",
    cleanup_output_dir: bool = True,
    n_trials: int = 20,
    sampler: optuna.samplers.BaseSampler | None = None,
) -> dict[str, tuple[TuningResult, list[TuningResult]]]:
    """
    Tune hyperparameters with Optuna for each group (e.g., GICS sectors) and global set.

    Returns:
        Dict mapping group name to (best_result, all_results).
    """
    groups = GICS_LEVEL_1 if groups is None else groups
    results: dict[str, tuple[TuningResult, list[TuningResult]]] = {}

    global_best, global_results = tune_hyperparams_for_dataframe_optuna(
        df=df,
        search_space=search_space,
        group_name="global",
        test_size=test_size,
        val_ratio=val_ratio,
        output_dir=output_dir,
        cleanup_output_dir=cleanup_output_dir,
        n_trials=n_trials,
        sampler=sampler,
    )
    results["global"] = (global_best, global_results)

    for group_name, tickers in groups.items():
        available = [ticker for ticker in tickers if ticker in df.columns]
        if not available:
            continue
        group_df = df[available]

        best, all_results = tune_hyperparams_for_dataframe_optuna(
            df=group_df,
            search_space=search_space,
            group_name=group_name,
            test_size=test_size,
            val_ratio=val_ratio,
            output_dir=output_dir,
            cleanup_output_dir=cleanup_output_dir,
            n_trials=n_trials,
            sampler=sampler,
        )
        results[group_name] = (best, all_results)
        _clear_memory()

    return results