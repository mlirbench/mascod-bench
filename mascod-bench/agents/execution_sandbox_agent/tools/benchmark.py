"""
tools/benchmark.py — IREE execution timing via the iree-benchmark-module CLI.

Drop alongside execution.py in your tools/ directory.

Key function: benchmark_vmfb()
  - Takes compiled flatbuffer bytes + input flags from mlir_utils.make_input_flags()
  - Shells out to iree-benchmark-module
  - Parses the Google Benchmark console output
  - Returns mean/std/min/p50/p90/p99 statistics

Why CLI and not Python bindings?
  iree-benchmark-module uses the Google Benchmark library internally and handles
  process isolation, CPU affinity, and warmup automatically. The Python runtime
  bindings do not expose equivalent timing infrastructure.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import Any

import numpy as np


# ── Unit normalisation ────────────────────────────────────────────────────────

def _to_ms(value: float, unit: str) -> float:
    return {
        "s":  value * 1_000.0,
        "ms": value,
        "us": value / 1_000.0,
        "ns": value / 1_000_000.0,
    }.get(unit, value)


# ── Google Benchmark output parser ────────────────────────────────────────────
#
# With --benchmark_repetitions=N and --benchmark_report_aggregates_only=false,
# iree-benchmark-module emits output like:
#
#   BM_RunModule/process_time/real_time          2.31 ms    2.33 ms    1
#   BM_RunModule/process_time/real_time          2.29 ms    2.31 ms    1
#   ...
#   BM_RunModule/process_time/real_time_mean     2.30 ms    2.32 ms    30
#   BM_RunModule/process_time/real_time_median   2.30 ms    2.31 ms    30
#   BM_RunModule/process_time/real_time_stddev   0.05 ms    0.05 ms    30
#   BM_RunModule/process_time/real_time_cv       2.17 %     2.16 %     30
#
# Individual repetition lines have no stat suffix.
# Aggregate lines end with _mean / _median / _stddev / _cv.

_AGGREGATE_SUFFIXES = ("_mean", "_median", "_stddev", "_cv")

_BM_LINE_RE = re.compile(
    r"^(BM_\S+/process_time/real_time\S*)\s+([\d.]+)\s+(ms|us|ns|s)",
    re.MULTILINE,
)


def _parse_benchmark_output(text: str) -> dict[str, Any]:
    raw_ms: list[float] = []
    agg: dict[str, float] = {}

    for m in _BM_LINE_RE.finditer(text):
        name, value_str, unit = m.group(1), m.group(2), m.group(3)
        value_ms = _to_ms(float(value_str), unit)

        matched_suffix = next((s for s in _AGGREGATE_SUFFIXES if name.endswith(s)), None)
        if matched_suffix:
            key = matched_suffix.lstrip("_")
            if key != "cv":
                agg[key] = value_ms
        else:
            raw_ms.append(value_ms)

    return {
        "raw_ms":     raw_ms,
        "agg_mean":   agg.get("mean"),
        "agg_stddev": agg.get("stddev"),
        "agg_median": agg.get("median"),
    }


def _compute_stats(
    raw_ms: list[float],
    agg_mean: float | None,
    agg_stddev: float | None,
) -> dict[str, Any]:
    if raw_ms:
        arr = np.array(raw_ms, dtype=np.float64)
        return {
            "mean_ms": round(float(agg_mean   if agg_mean   is not None else np.mean(arr)), 4),
            "std_ms":  round(float(agg_stddev if agg_stddev is not None else (np.std(arr, ddof=1) if len(arr) > 1 else 0.0)), 4),
            "min_ms":  round(float(np.min(arr)), 4),
            "p50_ms":  round(float(np.percentile(arr, 50)), 4),
            "p90_ms":  round(float(np.percentile(arr, 90)), 4),
            "p99_ms":  round(float(np.percentile(arr, 99)), 4),
        }

    # Fallback: only aggregate lines were parsed (older IREE builds)
    if agg_mean is not None:
        return {
            "mean_ms": round(agg_mean, 4),
            "std_ms":  round(agg_stddev, 4) if agg_stddev is not None else None,
            "min_ms":  None,
            "p50_ms":  None,
            "p90_ms":  None,
            "p99_ms":  None,
            "_note":   "p50/p90/p99 unavailable — individual rep lines not parsed",
        }

    return {}


# ── Public API ────────────────────────────────────────────────────────────────

def benchmark_vmfb(
    compiled_module: bytes,
    input_flags: list[str],
    target_backend: str = "llvm-cpu",
    iterations: int = 30,
    warmup_seconds: float = 10.0,
    timeout_seconds: int = 600,
    max_retries: int = 3,
) -> dict[str, Any]:
    """
    Benchmark a compiled IREE module using the iree-benchmark-module CLI.

    Args:
        compiled_module:  Flatbuffer bytes from compile_mlir()["compiled_module"].
        input_flags:      --input=<type> strings from mlir_utils.make_input_flags().
                          Required when @main takes tensor arguments.
        target_backend:   Must match the backend used to compile.
        iterations:       --benchmark_repetitions passed to iree-benchmark-module.
        warmup_seconds:   --benchmark_min_warmup_time (seconds). The tool runs
                          warmup until this duration is met before measuring.
        timeout_seconds:  Kill subprocess after this many seconds.
                          Default 600s covers 10s warmup + 30 reps comfortably.
        max_retries:      Number of attempts if iree-benchmark-module fails with
                          no benchmark output (e.g. non-deterministic exit 245).

    Returns:
        {
          "status":    "success" | "error"
          "mean_ms":   float
          "std_ms":    float
          "min_ms":    float
          "p50_ms":    float
          "p90_ms":    float
          "p99_ms":    float
          "raw_count": int     — number of individual rep lines parsed
        }
    """
    binary = shutil.which("iree-benchmark-module")
    if binary is None:
        return {
            "status": "error",
            "error_message": (
                "iree-benchmark-module not found on PATH. "
                "Ensure IREE tools are installed and on PATH."
            ),
        }

    driver = "local-task" if target_backend == "llvm-cpu" else target_backend

    vmfb_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".vmfb", delete=False) as f:
            f.write(compiled_module)
            vmfb_path = f.name

        cmd = [
            binary,
            f"--module={vmfb_path}",
            f"--device={driver}",
            "--function=main",
            f"--benchmark_repetitions={iterations}",
            f"--benchmark_min_warmup_time={warmup_seconds:.1f}",
            "--benchmark_report_aggregates_only=false",
            "--benchmark_format=console",
            *input_flags,
        ]

        last_proc = None
        parsed: dict[str, Any] = {"raw_ms": [], "agg_mean": None, "agg_stddev": None, "agg_median": None}
        for attempt in range(1, max_retries + 1):
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
            last_proc = proc

            # Google Benchmark writes timing lines to stdout; IREE writes warnings
            # to stderr. Parse stdout first so timing data isn't pushed past any
            # truncation threshold in error messages.
            output_text = (proc.stdout or "") + (proc.stderr or "")
            parsed = _parse_benchmark_output(output_text)

            if parsed["raw_ms"] or parsed["agg_mean"] is not None:
                break  # success — got benchmark data

            # No benchmark data; retry unless out of attempts
            if attempt < max_retries:
                continue

        # All attempts exhausted without benchmark data
        if not parsed["raw_ms"] and parsed["agg_mean"] is None:
            proc = last_proc  # type: ignore[assignment]
            return {
                "status": "error",
                "error_message": (
                    f"iree-benchmark-module produced no benchmark data after "
                    f"{max_retries} attempt(s) (exit={proc.returncode}). "
                    f"stdout: {proc.stdout[:1000]!r} | "
                    f"stderr: {proc.stderr[:2000]!r}"
                ),
            }

        stats = _compute_stats(
            raw_ms=parsed["raw_ms"],
            agg_mean=parsed["agg_mean"],
            agg_stddev=parsed["agg_stddev"],
        )

        return {
            "status":    "success",
            "raw_count": len(parsed["raw_ms"]),
            **stats,
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error_message": (
                f"iree-benchmark-module timed out after {timeout_seconds}s. "
                "Consider increasing --timeout or reducing --warmup."
            ),
        }
    except Exception as exc:
        return {"status": "error", "error_message": str(exc)}
    finally:
        if vmfb_path and os.path.exists(vmfb_path):
            try:
                os.unlink(vmfb_path)
            except Exception:
                pass
