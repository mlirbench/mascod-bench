"""
mlir_embed.py — Embed deterministic inputs into MLIR programs.

Transforms programs with tensor arguments into self-contained programs
by embedding constants derived from:
  1. Static analysis of the program's own constants (program-specific)
  2. A generic boundary-covering pattern (structural fallback)
"""

from __future__ import annotations
import re
from functools import reduce


# ── Generic fallback patterns ─────────────────────────────────────────────────
# Used when no program-specific constants are found, or to pad remaining slots.

_FLOAT_BASE = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1.5, -1.5, 0.25, -0.25]
_INT_BASE   = [0,   1,   -1,   2,   -2,   3,   -3,   4,   -4,   5,   -5  ]
_BOOL_BASE  = [False, True, False, True, False, True, False, True, False, True]


def extract_constants_from_mlir(mlir_source: str) -> list[float]:
    """
    Extracts scalar numeric constants from arith.constant lines (linalg dialect)
    and torch.constant.float / torch.vtensor.literal lines (torch dialect).

    Handles:
      arith.constant 4.0 : f32                         → [4.0]
      arith.constant dense<0.5>                         → [0.5]
      arith.constant dense<[1.0, 2.0]>                  → [1.0, 2.0]
      arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]>   → [1.0, 2.0, 3.0, 4.0]
      torch.constant.float 0.000000e+00                 → [0.0]
      torch.vtensor.literal(dense<9.0> : tensor<f32>)  → [9.0]
    Skips: true/false, non-numeric literals
    """
    values: list[float] = []

    # ── Linalg: arith.constant ───────────────────────────────────────────────
    for match in re.finditer(r'arith\.constant\s+(.*?)\s*:', mlir_source):
        raw = match.group(1).strip()

        # Skip boolean literals
        if raw in {"true", "false"}:
            continue

        # dense<...> literal — extract all numbers inside
        dense_match = re.search(r'dense<([^>]+)>', raw)
        if dense_match:
            inner = dense_match.group(1)
            for num in re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', inner):
                try:
                    values.append(float(num))
                except ValueError:
                    continue
        else:
            # Scalar constant: arith.constant 4.0 : f32
            for num in re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', raw):
                try:
                    values.append(float(num))
                except ValueError:
                    continue

    # ── Torch: torch.constant.float ──────────────────────────────────────────
    for match in re.finditer(r'torch\.constant\.float\s+([\d.e+\-]+)', mlir_source):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue

    # ── Torch: torch.vtensor.literal(dense<...>) ─────────────────────────────
    for match in re.finditer(r'torch\.vtensor\.literal\s*\(\s*dense<([^>]+)>', mlir_source):
        for num in re.findall(r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?', match.group(1)):
            try:
                values.append(float(num))
            except ValueError:
                continue

    # Deduplicate while preserving order
    seen = set()
    result = []
    for v in values:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result


def build_neighborhood(values: list[float], dtype: str) -> list[float]:
    """
    For each value k, adds k-1, k-0.1, k, k+0.1, k+1.
    Always includes 0.0 as a baseline.
    For integer dtypes, uses integer steps only.
    """
    neighborhood: list[float] = [0.0]

    for k in values:
        if dtype in {"i32", "i64", "i1"}:
            neighborhood.extend([k - 1, k, k + 1])
        else:
            neighborhood.extend([k - 1.0, k - 0.1, k, k + 0.1, k + 1.0])

    # Deduplicate while preserving order
    seen = set()
    result = []
    for v in neighborhood:
        rounded = round(v, 6)
        if rounded not in seen:
            seen.add(rounded)
            result.append(v)
    return result


def parse_tensor_type(type_str: str) -> tuple[list[int], str]:
    """
    "tensor<3x3xf32>"            → ([3, 3], "f32")
    "tensor<4xi32>"              → ([4],    "i32")
    "tensor<8xi1>"               → ([8],    "i1")
    "tensor<f32>"                → ([],     "f32")  scalar
    "!torch.vtensor<[3,3],f32>"  → ([3, 3], "f32")
    """
    # Torch dialect
    torch_match = re.search(r'vtensor<\[([^\]]*)\],\s*(\w+)>', type_str)
    if torch_match:
        dims_str = torch_match.group(1)
        dtype    = torch_match.group(2)
        shape = [int(d) for d in dims_str.split(',') if d.strip().isdigit()]
        return shape, dtype

    # Linalg dialect: tensor<3x3xf32> or tensor<f32>
    linalg_match = re.search(r'tensor<([^>]+)>', type_str)
    if linalg_match:
        inner = linalg_match.group(1)
        parts = inner.split('x')
        shape = []
        dtype = parts[-1]  # last part is always the dtype
        for part in parts[:-1]:
            if part.isdigit():
                shape.append(int(part))
        return shape, dtype

    return [], "f32"  # fallback


def generate_input_values(
    mlir_source: str,
    shape: list[int],
    dtype: str,
) -> list:
    """
    Returns a flat list of values for the given shape and dtype.
    Combines extracted program constants with a generic fallback pattern.
    """
    # Total number of elements needed
    total = reduce(lambda a, b: a * b, shape, 1) if shape else 1

    # Step 1: extract program-specific constants
    extracted = extract_constants_from_mlir(mlir_source)
    neighborhood = build_neighborhood(extracted, dtype)

    # Step 2: select the right base pattern for this dtype
    if dtype == "i1":
        base = _BOOL_BASE
    elif dtype in {"i32", "i64", "i8", "i16", "si32", "si64"}:
        base = _INT_BASE
    else:
        base = _FLOAT_BASE

    # Step 3: merge — program-specific values take priority, base fills the rest
    combined = neighborhood + [v for v in base if v not in set(neighborhood)]

    # Step 4: cast to the right Python type
    if dtype == "i1":
        flat = [bool(v) for v in combined]
    elif dtype in {"i32", "i64", "i8", "i16", "si32", "si64"}:
        flat = [int(round(v)) for v in combined]
    else:
        flat = [float(v) for v in combined]

    # Step 5: cycle to fill total elements
    return [flat[i % len(flat)] for i in range(total)]


def _format_value(v, dtype: str) -> str:
    if dtype == "i1":
        return "true" if v else "false"
    if dtype in {"i32", "i64", "i8", "i16", "si32", "si64"}:
        return str(int(v))
    # Float: MLIR dense<> requires a decimal point in every float literal.
    # Python's repr() can produce "1e-05" (no decimal) which iree-compile rejects.
    # Use %g-style but guarantee a decimal point is present.
    if v == int(v) and abs(v) < 1e6:
        return f"{v:.1f}"
    s = repr(float(v))
    # Ensure there is a '.' before any 'e' so MLIR parses it as a float literal
    if 'e' in s and '.' not in s.split('e')[0]:
        s = s.replace('e', '.0e', 1)
    return s


def _nest(flat: list, shape: list[int]):
    """Recursively reshape a flat list into nested lists."""
    if len(shape) == 1:
        return flat[:shape[0]]
    stride = reduce(lambda a, b: a * b, shape[1:], 1)
    return [_nest(flat[i*stride:(i+1)*stride], shape[1:]) for i in range(shape[0])]


def _format_nested(nested, dtype: str) -> str:
    if isinstance(nested, list):
        inner = ", ".join(_format_nested(v, dtype) for v in nested)
        return f"[{inner}]"
    return _format_value(nested, dtype)


def _linalg_tensor_type(shape: list[int], dtype: str) -> str:
    """Build a standard linalg tensor type string from shape + dtype, e.g. 'tensor<3x3xf32>'."""
    if not shape:
        return f"tensor<{dtype}>"
    return "tensor<" + "x".join(str(d) for d in shape) + f"x{dtype}>"


def format_mlir_dense(flat: list, shape: list[int], dtype: str, type_str: str) -> str:
    """
    Returns the full constant line for the given tensor type, e.g.:
      arith.constant dense<[[1.0, -1.0], [0.0, 0.5]]> : tensor<2x2xf32>          (linalg)
      torch.vtensor.literal(dense<[[1.0, -1.0]]> : tensor<1x2xf32>) : !torch.vtensor<[1,2],f32>  (torch)
    """
    if not shape:
        val = _format_value(flat[0], dtype)
        body = val
    else:
        nested = _nest(flat, shape)
        body   = _format_nested(nested, dtype)

    # Torch vtensor: use torch.vtensor.literal(dense<...> : tensor<...>) : !torch.vtensor<...>
    if "vtensor<" in type_str:
        inner_type = _linalg_tensor_type(shape, dtype)
        return f"torch.vtensor.literal(dense<{body}> : {inner_type}) : {type_str}"

    # Linalg / standard tensor
    return f"arith.constant dense<{body}> : {type_str}"


def parse_main_signature(mlir_source: str) -> list[tuple[str, str]]:
    """
    Returns list of (name, type_str) pairs, e.g.:
    [("%arg0", "tensor<3x3xf32>"), ("%arg1", "tensor<4xi32>")]
    Returns [] if @main takes no arguments.

    Handles argument names that may not follow the %argN convention
    (e.g. mutations may rename %arg0 to %cst or other SSA values).
    Only includes arguments whose type is a tensor or vtensor (not scalars
    like i32 / f32 that IREE cannot accept as top-level inputs).
    """
    sig_match = re.search(r'func\.func\s+@main\s*\(([^)]*)\)', mlir_source)
    if not sig_match:
        return []

    args_str = sig_match.group(1).strip()
    if not args_str:
        return []

    # Split on commas that are not inside angle brackets (handles tensor<...>)
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in args_str:
        if ch == '<':
            depth += 1
            current.append(ch)
        elif ch == '>':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append(''.join(current))

    args = []
    for part in parts:
        m = re.match(r'\s*(%\w+)\s*:\s*(.+)', part.strip())
        if not m:
            continue
        name     = m.group(1).strip()
        type_str = m.group(2).strip()
        # Only embed tensor-typed arguments; scalar args are not embeddable
        if re.search(r'tensor<|vtensor<', type_str):
            args.append((name, type_str))
    return args


def _safe_constant_name(name: str, mlir_source: str) -> str:
    """
    Returns `name` unchanged if it is not already defined in the body,
    otherwise returns a collision-free alternative like %embedded_cst_0.

    In valid MLIR, a function argument and a body definition cannot share
    the same SSA name (SSA uniqueness). This guard handles the degenerate
    case where a malformed mutation creates exactly that situation.
    """
    body_defines = set(re.findall(r'(%\w+)\s*=', mlir_source))
    if name not in body_defines:
        return name
    base = name.lstrip('%')
    i = 0
    while f"%embedded_{base}_{i}" in body_defines:
        i += 1
    return f"%embedded_{base}_{i}"


def embed_inputs(mlir_source: str) -> tuple[str, dict]:
    """
    Transforms an MLIR program with tensor arguments into a
    self-contained program with embedded constants.

    Returns (transformed_mlir, embedded_inputs_dict) where
    embedded_inputs_dict maps each original argument name to its nested
    value list, e.g. {"%arg0": [[0.0, 8.0], ...]}.

    Returns (mlir_source, {}) unchanged if @main already takes no arguments.
    """
    args = parse_main_signature(mlir_source)
    if not args:
        return mlir_source, {}  # already self-contained

    # Build one constant line per argument; track any name collisions
    renames: dict[str, str] = {}
    constant_lines = []
    inputs_dict: dict = {}
    for name, type_str in args:
        shape, dtype = parse_tensor_type(type_str)
        flat         = generate_input_values(mlir_source, shape, dtype)
        const_expr   = format_mlir_dense(flat, shape, dtype, type_str)
        safe_name    = _safe_constant_name(name, mlir_source)
        constant_lines.append(f"    {safe_name} = {const_expr}")
        if safe_name != name:
            renames[name] = safe_name
        # Store as nested list under the original argument name
        inputs_dict[name] = _nest(flat, shape) if shape else flat[0]

    constants_block = "\n".join(constant_lines)

    # Change 1: remove arguments from function signature
    result = re.sub(
        r'(func\.func\s+@main\s*)\([^)]*\)',
        r'\1()',
        mlir_source,
    )

    # Change 2: insert constants immediately after the opening brace
    result = re.sub(
        r'(func\.func\s+@main[^{]*\{)',
        r'\1\n' + constants_block + '\n',
        result,
    )

    # Change 3: replace renamed argument *uses* throughout the body.
    # Negative lookahead (?!\w) prevents clobbering %cst_0 / %cst_something;
    # negative lookahead (?!\s*=) prevents rewriting the definition itself.
    for old_name, new_name in renames.items():
        result = re.sub(re.escape(old_name) + r'(?!\w)(?!\s*=)', new_name, result)

    return result, inputs_dict
