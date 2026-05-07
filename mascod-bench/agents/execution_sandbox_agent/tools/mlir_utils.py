"""
tools/mlir_utils.py — MLIR source parsing utilities.

Provides two things needed by the benchmark pipeline:
  1. parse_input_spec()  — extracts tensor shapes/dtypes from @main's arg list
  2. detect_ir_type()    — classifies the dominant dialect in the source
  3. make_input_flags()  — converts input_spec into iree-benchmark-module --input= flags
"""

from __future__ import annotations

import re
from typing import Any


# ── Tensor type parser ────────────────────────────────────────────────────────

# Matches the entire argument list of func.func @main(...)
# Stops at the first ')' which closes the arg list (not the return type).
_FUNC_SIG_RE = re.compile(
    r"func\.func\s+@main\s*\(([^)]*)\)",
    re.DOTALL,
)

# Matches a linalg tensor type string inside tensor<...>
# Negative lookbehind prevents matching vtensor< (torch dialect).
# Captures the contents, e.g. "3x3xf32" or "i1" or "3xi1"
_TENSOR_TYPE_RE = re.compile(r"(?<![a-zA-Z])tensor<([^>]+)>")

# Matches a torch vtensor type: !torch.vtensor<[3,3],f32>
# Captures the bracketed shape and the dtype separately.
_TORCH_VTENSOR_RE = re.compile(r"!torch\.vtensor<\[([^\]]*)\],\s*([^>]+)>")


def _parse_tensor_type_str(type_str: str) -> dict[str, Any]:
    """
    Convert an MLIR tensor content string into a shape/dtype dict.

    Examples:
        "3x3xf32"     -> {"shape": [3, 3], "dtype": "f32"}
        "1024x1024xf32" -> {"shape": [1024, 1024], "dtype": "f32"}
        "3xf64"       -> {"shape": [3], "dtype": "f64"}
        "i1"          -> {"shape": [], "dtype": "i1"}   (0-d / scalar tensor)
        "f32"         -> {"shape": [], "dtype": "f32"}
    """
    parts = type_str.split("x")
    shape: list[int] = []
    dtype: str = type_str  # fallback if nothing parses

    for i, part in enumerate(parts):
        if part.isdigit():
            shape.append(int(part))
        else:
            # Everything from here onward is the dtype
            # (handles types like "bf16", "f32", "i32", "i1", etc.)
            dtype = "x".join(parts[i:])
            break

    return {"shape": shape, "dtype": dtype}


def parse_input_spec(mlir_source: str) -> list[dict[str, Any]]:
    """
    Parse the @main function signature and return one dict per tensor argument.

    Returns a list because some programs (e.g. program_3.mlir) have multiple
    input tensors. Returns an empty list if the signature cannot be found or
    contains no tensor arguments.

    Handles both:
      - linalg dialect:  tensor<3x3xf32>
      - torch dialect:   !torch.vtensor<[3,3],f32>

    Example output for program_1.mlir (linalg):
        [{"shape": [3, 3], "dtype": "f32"}]

    Example output for program_3.mlir (linalg, two inputs):
        [{"shape": [3, 3], "dtype": "f32"}, {"shape": [3, 3], "dtype": "f32"}]
    """
    m = _FUNC_SIG_RE.search(mlir_source)
    if not m:
        return []

    args_str = m.group(1)

    # Linalg path: tensor<3x3xf32>
    linalg_types = _TENSOR_TYPE_RE.findall(args_str)
    if linalg_types:
        return [_parse_tensor_type_str(t) for t in linalg_types]

    # Torch path: !torch.vtensor<[3,3],f32>
    torch_types = _TORCH_VTENSOR_RE.findall(args_str)
    result = []
    for shape_str, dtype in torch_types:
        shape_str = shape_str.strip()
        if shape_str:
            shape = [int(d) for d in shape_str.split(",") if d.strip().lstrip("-").isdigit()]
        else:
            shape = []
        result.append({"shape": shape, "dtype": dtype.strip()})
    return result


# ── IR type detector ──────────────────────────────────────────────────────────

# Priority-ordered dialect signatures.
# First match wins, so more specific dialects go first.
_DIALECT_SIGNATURES: list[tuple[str, list[str]]] = [
    ("torch",  ["torch.", "torch_c."]),
    ("affine", ["affine.for", "affine.load", "affine.store", "affine.apply"]),
    ("linalg", ["linalg.generic", "linalg.matmul", "linalg.fill", "linalg.conv"]),
]


def detect_ir_type(mlir_source: str) -> str:
    """
    Return the dominant dialect in an MLIR program.

    Checks for dialect-specific op prefixes in priority order:
      torch  > affine  > linalg

    Falls back to "unknown" if nothing matches.
    """
    for ir_type, signatures in _DIALECT_SIGNATURES:
        if any(sig in mlir_source for sig in signatures):
            return ir_type
    return "unknown"


# ── iree-benchmark-module input flag builder ──────────────────────────────────

def _type_str_for_iree(spec: dict[str, Any]) -> str:
    """
    Convert a parsed input_spec dict to an IREE type string.

    Examples:
        {"shape": [3, 3], "dtype": "f32"}  -> "3x3xf32"
        {"shape": [3],    "dtype": "i1"}   -> "3xi1"
        {"shape": [],     "dtype": "f32"}  -> "f32"   (scalar / 0-d tensor)
    """
    shape = spec.get("shape", [])
    dtype = spec.get("dtype", "f32")
    if shape:
        return "x".join(str(d) for d in shape) + "x" + dtype
    return dtype


def make_input_flags(input_specs: list[dict[str, Any]]) -> list[str]:
    """
    Return a list of --input=<type> flags for iree-benchmark-module.

    Passing a type string without a value causes IREE to fill the tensor
    with zeros, which is fine for timing-only benchmarks.

    Example:
        [{"shape": [3, 3], "dtype": "f32"}]
        -> ["--input=3x3xf32"]

        [{"shape": [3, 3], "dtype": "f32"}, {"shape": [3, 3], "dtype": "f32"}]
        -> ["--input=3x3xf32", "--input=3x3xf32"]
    """
    return [f"--input={_type_str_for_iree(spec)}" for spec in input_specs]
