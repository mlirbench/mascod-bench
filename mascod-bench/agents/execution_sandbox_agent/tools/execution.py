"""Execution tools for compiling and running MLIR programs with IREE."""

from __future__ import annotations

import ast
import base64
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np


def _import_iree_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        short_name = module_name.rsplit(".", 1)[-1]
        raise ModuleNotFoundError(
            "Missing IREE %s dependency. Install a version of the IREE Python "
            "packages that supports the active Python interpreter and platform."
            % short_name
        ) from exc


def _candidate_input_types(dialect: str) -> list[str]:
    base_candidates = ["auto"]
    if dialect == "torch":
        base_candidates.extend(["torch", "tm_tensor"])

    seen: set[str] = set()
    ordered_candidates: list[str] = []
    for candidate in base_candidates:
        if candidate not in seen:
            ordered_candidates.append(candidate)
            seen.add(candidate)
    return ordered_candidates


def _driver_name_for_backend(target_backend: str) -> str:
    if target_backend == "llvm-cpu":
        return "local-task"
    return target_backend


def _compile_extra_args(target_backend: str) -> list[str]:
    if target_backend == "llvm-cpu":
        return [
            "--iree-llvmcpu-target-cpu=generic",
            "--iree-llvmcpu-loop-vectorization=false",
            "--iree-llvmcpu-slp-vectorization=false",
            "--iree-llvmcpu-loop-interleaving=false",
            "--iree-llvmcpu-loop-unrolling=false",
        ]
    return []


_MLIR_DTYPE_MAP: dict[str, str] = {
    "f16":  "float16",
    "f32":  "float32",
    "f64":  "float64",
    "i1":   "bool",
    "i8":   "int8",
    "i16":  "int16",
    "i32":  "int32",
    "i64":  "int64",
    "si32": "int32",   # torch signed int
    "si64": "int64",   # torch signed int
    "ui8":  "uint8",
    "ui16": "uint16",
    "ui32": "uint32",
    "ui64": "uint64",
    "bf16": "float32",
}


def _split_func_args(args_str: str) -> list[str]:
    """Split function arg list by commas, respecting nested angle brackets."""
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in args_str:
        if ch == "<":
            depth += 1
            current.append(ch)
        elif ch == ">":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        parts.append("".join(current))
    return parts


def _parse_mlir_type(type_str: str) -> dict | None:
    """Convert an MLIR type string to {'shape': [...], 'dtype': '<numpy dtype name>'}."""
    type_str = type_str.strip()
    # Torch dialect: !torch.vtensor<[3,3],f32>
    torch_match = re.search(r"vtensor<\[([^\]]*)\],\s*(\w+)>", type_str)
    if torch_match:
        dims_str = torch_match.group(1)
        elem_type = torch_match.group(2)
        shape = [int(d) for d in dims_str.split(",") if d.strip().isdigit()]
        dtype = _MLIR_DTYPE_MAP.get(elem_type)
        if dtype is None:
            return None
        return {"shape": shape, "dtype": dtype}
    # Linalg dialect: tensor<3x3xf32>
    tensor_match = re.fullmatch(r"tensor<([^>]+)>", type_str)
    if tensor_match:
        inner = tensor_match.group(1)
        parts = inner.split("x")
        elem_type = parts[-1]
        shape = [int(d) for d in parts[:-1] if d.isdigit()]
        dtype = _MLIR_DTYPE_MAP.get(elem_type)
        if dtype is None:
            return None
        return {"shape": shape, "dtype": dtype}
    scalar_dtype = _MLIR_DTYPE_MAP.get(type_str)
    if scalar_dtype is not None:
        return {"shape": [], "dtype": scalar_dtype}
    return None


def parse_input_spec(mlir_source: str) -> list[dict]:
    """
    Parse the @main function signature and return a list of input specs.
    Each spec is {'shape': list[int], 'dtype': str} where dtype is a numpy dtype name.
    """
    match = re.search(r"func\.func\s+@main\s*\(([^)]*)\)", mlir_source)
    if not match:
        return []
    args_str = match.group(1).strip()
    if not args_str:
        return []
    specs: list[dict] = []
    for part in _split_func_args(args_str):
        colon_idx = part.find(":")
        if colon_idx == -1:
            continue
        spec = _parse_mlir_type(part[colon_idx + 1:])
        if spec is not None:
            specs.append(spec)
    return specs


def _generate_deterministic_inputs(input_specs: list[dict], seed: int = 42) -> list[np.ndarray]:
    """Generate deterministic random inputs from parsed specs using a fixed seed."""
    rng = np.random.default_rng(seed)
    inputs: list[np.ndarray] = []
    for spec in input_specs:
        shape = tuple(spec["shape"])
        dtype_str = spec["dtype"]
        if dtype_str == "bool":
            arr = rng.integers(0, 2, size=shape or (1,), dtype=np.uint8).astype(np.bool_)
            if not shape:
                arr = arr.reshape(())
        elif dtype_str in ("float16", "float32", "float64"):
            arr = rng.standard_normal(shape or (1,)).astype(np.dtype(dtype_str))
            if not shape:
                arr = arr.reshape(())
        else:
            arr = rng.integers(-100, 100, size=shape or (1,)).astype(np.dtype(dtype_str))
            if not shape:
                arr = arr.reshape(())
        inputs.append(arr)
    return inputs


def _serialize_inputs(inputs: list[np.ndarray]) -> list[dict]:
    """Serialize numpy arrays to JSON-safe dicts for subprocess transfer."""
    return [
        {"dtype": str(inp.dtype), "shape": list(inp.shape), "data": inp.tolist()}
        for inp in inputs
    ]


def _normalize_result(value: Any) -> Any:
    if hasattr(value, "to_host"):
        try:
            value = value.to_host()
        except Exception:
            pass

    if isinstance(value, tuple):
        return tuple(_normalize_result(item) for item in value)
    if isinstance(value, list):
        return [_normalize_result(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize_result(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()

    try:
        array_value = np.asarray(value)
    except Exception:
        return value

    if array_value.shape == ():
        try:
            return array_value.item()
        except Exception:
            return value

    if array_value.dtype.kind in {"b", "i", "u", "f", "c"}:
        return array_value.tolist()

    return value


def _serialize_output(value: Any) -> str:
    return repr(_normalize_result(value))


def _normalize_runtime_error_message(error_message: str) -> str:
    argument_mismatch = re.search(
        r"expected\s+(\d+)\s+arguments\s+but\s+passed\s+(\d+)",
        error_message,
        flags=re.IGNORECASE,
    )
    if argument_mismatch:
        expected_args = argument_mismatch.group(1)
        passed_args = argument_mismatch.group(2)
        return (
            "Exported 'main' requires input arguments and cannot be run with the "
            f"current zero-argument execution path: expected {expected_args}, passed {passed_args}"
        )
    return error_message


def _extract_main_callable(module_obj: Any):
    if module_obj is None:
        return None

    if hasattr(module_obj, "__getitem__"):
        try:
            candidate = module_obj["main"]
            if callable(candidate):
                return candidate
        except Exception:
            pass

    candidate = getattr(module_obj, "main", None)
    if callable(candidate):
        return candidate

    return None


def _call_main_function(main_fn: Any, inputs: list[Any] | None = None) -> Any:
    args = list(inputs) if inputs else []
    try:
        return main_fn(*args)
    except TypeError as exc:
        signature = None
        try:
            signature = inspect.signature(main_fn)
        except Exception:
            signature = None

        if signature is not None:
            raise RuntimeError(
                f"Exported 'main' requires arguments and cannot be run with no inputs: {signature}"
            ) from exc

        raise RuntimeError(
            "Exported 'main' requires arguments and cannot be run with no inputs"
        ) from exc


def _invoke_main_via_loaded_module(runtime_module: Any, compiled_module: bytes, driver_name: str, inputs: list[Any] | None = None) -> Any:
    load_vm_flatbuffer = getattr(runtime_module, "load_vm_flatbuffer", None)
    if not callable(load_vm_flatbuffer):
        raise RuntimeError("iree.runtime.load_vm_flatbuffer is not available")

    bound_module = None
    load_attempts = [
        {"vm_flatbuffer": compiled_module, "driver": driver_name},
        {"vm_flatbuffer": compiled_module, "driver_name": driver_name},
        {"flatbuffer": compiled_module, "driver": driver_name},
        {"flatbuffer": compiled_module, "driver_name": driver_name},
    ]
    for kwargs in load_attempts:
        try:
            bound_module = load_vm_flatbuffer(**kwargs)
            break
        except TypeError:
            continue

    if bound_module is None:
        try:
            bound_module = load_vm_flatbuffer(compiled_module, driver=driver_name)
        except TypeError:
            bound_module = load_vm_flatbuffer(compiled_module)

    main_fn = _extract_main_callable(bound_module)
    if main_fn is None:
        raise RuntimeError("Could not find exported 'main' function in loaded VM module")

    return _call_main_function(main_fn, inputs)


def _invoke_main_via_system_context(runtime_module: Any, compiled_module: bytes, driver_name: str, inputs: list[Any] | None = None) -> Any:
    config = runtime_module.Config(driver_name=driver_name)
    vm_instance = getattr(config, "vm_instance", None) or getattr(config, "instance", None)
    vm_module_cls = getattr(runtime_module, "VmModule", None)

    if vm_instance is None or vm_module_cls is None:
        raise RuntimeError("IREE runtime does not expose VmModule creation APIs")

    if hasattr(vm_module_cls, "copy_buffer"):
        vm_module = vm_module_cls.copy_buffer(vm_instance, compiled_module)
    elif hasattr(vm_module_cls, "from_flatbuffer"):
        vm_module = vm_module_cls.from_flatbuffer(vm_instance, compiled_module)
    else:
        raise RuntimeError("IREE runtime VmModule API is unsupported in this environment")

    try:
        context = runtime_module.SystemContext(config=config, vm_modules=(vm_module,))
    except TypeError:
        context = runtime_module.SystemContext(config=config)
        add_vm_module = getattr(context, "add_vm_module", None)
        if not callable(add_vm_module):
            raise RuntimeError("IREE SystemContext does not allow loading VmModule")
        add_vm_module(vm_module)

    modules = getattr(context, "modules", None)
    if modules is None:
        raise RuntimeError("IREE SystemContext did not expose loaded modules")

    main_fn = _extract_main_callable(getattr(modules, "module", None))
    if main_fn is None:
        for attribute_name in dir(modules):
            if attribute_name.startswith("_"):
                continue
            main_fn = _extract_main_callable(getattr(modules, attribute_name, None))
            if main_fn is not None:
                break

    if main_fn is None:
        raise RuntimeError("Could not find exported 'main' function in SystemContext")

    return _call_main_function(main_fn, inputs)


def _run_compiled_from_file(flatbuffer_path: str, driver_name: str, inputs_file: str | None = None) -> dict:
    try:
        runtime_module = _import_iree_module("iree.runtime")
        with open(flatbuffer_path, "rb") as compiled_file:
            compiled_module = compiled_file.read()

        inputs: list[Any] = []
        if inputs_file:
            with open(inputs_file, encoding="utf-8") as f:
                raw_inputs = json.load(f)
            inputs = [np.array(item["data"], dtype=item["dtype"]) for item in raw_inputs]

        try:
            result = _invoke_main_via_loaded_module(runtime_module, compiled_module, driver_name, inputs)
        except Exception:
            result = _invoke_main_via_system_context(runtime_module, compiled_module, driver_name, inputs)

        return {"status": "success", "output": _normalize_result(result)}
    except Exception as exc:
        return {
            "status": "error",
            "error_message": _normalize_runtime_error_message(str(exc)),
        }


def _parse_output_value(output_text: str) -> Any:
    stripped = output_text.strip()
    if stripped == "":
        return ""

    if stripped == "None":
        return None

    try:
        return ast.literal_eval(stripped)
    except Exception:
        pass

    if stripped.lower() in {"true", "false"}:
        return stripped.lower() == "true"

    scalar_candidate = np.fromstring(stripped, sep=" ")
    if scalar_candidate.size == 1 and scalar_candidate.dtype.kind in {"i", "u", "f"}:
        return scalar_candidate[0].item()

    bracketed = stripped.replace("\n", " ")
    bracketed = re.sub(r"\]\s+\[", "], [", bracketed)
    bracketed = re.sub(r"(?<=[0-9eE\.\-\+])\s+(?=[\-\+0-9])", ", ", bracketed)
    try:
        return ast.literal_eval(bracketed)
    except Exception:
        return stripped


def _to_numpy_if_numeric(value: Any) -> np.ndarray | None:
    if value is None or isinstance(value, str):
        return None

    try:
        array_value = np.asarray(value)
    except Exception:
        return None

    if array_value.dtype.kind in {"b", "i", "u", "f", "c"}:
        return array_value

    return None


def _trim_compile_error(error_text: str) -> str:
    """Strip IREE compiler invocation details from error messages, keeping only diagnostics."""
    for marker in ("\nInvoked with:", "\nNeed more information?", "Need more information?"):
        idx = error_text.find(marker)
        if idx != -1:
            error_text = error_text[:idx].rstrip()
    # Strip Apple Silicon LLVM CPU-features noise — it masks the real MLIR diagnostic
    lines = error_text.splitlines()
    lines = [l for l in lines if not l.startswith("Internal error while creating host target:")]
    error_text = "\n".join(lines).strip()
    return error_text


def _compile_mlir_worker(mlir_file: str, dialect: str, target_backend: str, result_file: str) -> None:
    """In-subprocess compile worker: reads MLIR from mlir_file, writes JSON result to result_file."""
    import json as _json

    try:
        compiler_module = _import_iree_module("iree.compiler")
    except ModuleNotFoundError as exc:
        with open(result_file, "w", encoding="utf-8") as f:
            _json.dump({"status": "error", "error_message": str(exc)}, f)
        return

    compile_str = getattr(compiler_module, "compile_str", None)
    if not callable(compile_str):
        with open(result_file, "w", encoding="utf-8") as f:
            _json.dump({"status": "error", "error_message": "IREE compiler does not expose compile_str"}, f)
        return

    with open(mlir_file, encoding="utf-8") as f:
        mlir_code = f.read()

    errors: list[str] = []
    compiled = None
    tried_without_input_type = False

    for input_type in _candidate_input_types(dialect):
        try:
            c = compile_str(
                mlir_code,
                target_backends=[target_backend],
                input_type=input_type,
                extra_args=_compile_extra_args(target_backend),
            )
            compiled = c
            break
        except TypeError as exc:
            if "input_type" in str(exc) and not tried_without_input_type:
                tried_without_input_type = True
                try:
                    c = compile_str(
                        mlir_code,
                        target_backends=[target_backend],
                        extra_args=_compile_extra_args(target_backend),
                    )
                    compiled = c
                    break
                except Exception as retry_exc:
                    errors.append(f"default compile attempt failed: {_trim_compile_error(str(retry_exc))}")
            else:
                errors.append(f"input_type={input_type}: {_trim_compile_error(str(exc))}")
        except Exception as exc:
            errors.append(f"input_type={input_type}: {_trim_compile_error(str(exc))}")

    if compiled is not None:
        result: dict = {
            "status": "success",
            "compiled_module_b64": base64.b64encode(compiled).decode("ascii"),
        }
    else:
        detail = "; ".join(errors) if errors else "Unknown compilation failure"
        result = {"status": "error", "error_message": detail}

    with open(result_file, "w", encoding="utf-8") as f:
        _json.dump(result, f)


def compile_mlir(
    mlir_code: str,
    dialect: str = "linalg",
    target_backend: str = "llvm-cpu"
) -> dict:
    """
    Compile an MLIR program string using IREE.

    Args:
        mlir_code: Full MLIR program as a string.
        dialect: Either "linalg" or "torch". May affect compilation flags.
        target_backend: IREE target backend (default: "llvm-cpu").

    Returns:
        dict with keys:
            - "status": "success" or "error"
            - "compiled_module": bytes (the compiled flatbuffer) if success
            - "error_message": str if error
    """
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    child_env = os.environ.copy()
    existing_pythonpath = child_env.get("PYTHONPATH")
    if existing_pythonpath:
        child_env["PYTHONPATH"] = os.pathsep.join([project_root, existing_pythonpath])
    else:
        child_env["PYTHONPATH"] = project_root

    runner_code = (
        "from execution_sandbox_agent.tools.execution import _compile_mlir_worker; "
        "import sys; "
        "_compile_mlir_worker(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])"
    )

    mlir_path: str | None = None
    result_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".mlir", delete=False, encoding="utf-8"
        ) as mlir_file:
            mlir_file.write(mlir_code)
            mlir_path = mlir_file.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as result_file:
            result_path = result_file.name

        try:
            subprocess.run(
                [sys.executable, "-c", runner_code,
                 mlir_path, dialect, target_backend, result_path],
                cwd=project_root,
                env=child_env,
                capture_output=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired:
            return {"status": "error", "error_message": "Compilation exceeded 120 seconds"}

        if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
            with open(result_path, encoding="utf-8") as f:
                result = json.load(f)
            if result.get("status") == "success":
                compiled_bytes = base64.b64decode(result["compiled_module_b64"])
                return {"status": "success", "compiled_module": compiled_bytes}
            return {
                "status": "error",
                "error_message": result.get("error_message", "Unknown compilation error"),
            }
        return {"status": "error", "error_message": "Compilation subprocess produced no output"}
    except Exception as exc:
        return {"status": "error", "error_message": str(exc)}
    finally:
        for path in [mlir_path, result_path]:
            if path:
                try:
                    os.unlink(path)
                except Exception:
                    pass


def run_compiled(
    compiled_module: bytes,
    mlir_source: str | None = None,
    target_backend: str = "llvm-cpu",
    timeout_seconds: int = 30,
    input_seed: int = 42,
) -> dict:
    """
    Execute a compiled IREE module and capture its output.

    If mlir_source is provided, the @main input signature is parsed and
    deterministic random inputs are generated with the given seed (default 42)
    and forwarded to the compiled function. Programs with no inputs are called
    with zero arguments as before.

    Args:
        compiled_module: Compiled IREE flatbuffer bytes from compile_mlir.
        mlir_source: Original MLIR source string used to infer input types/shapes.
        target_backend: Must match the backend used during compilation.
        timeout_seconds: Maximum execution time before killing the process.
        input_seed: RNG seed for deterministic input generation (default 42).

    Returns:
        dict with keys:
            - "status": "success" or "error" or "timeout"
            - "output": raw Python value(s) from _normalize_result() if success
            - "error_message": str if error/timeout
    """
    try:
        _import_iree_module("iree.runtime")
    except ModuleNotFoundError as exc:
        return {"status": "error", "error_message": str(exc)}

    driver_name = _driver_name_for_backend(target_backend)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    child_env = os.environ.copy()
    existing_pythonpath = child_env.get("PYTHONPATH")
    if existing_pythonpath:
        child_env["PYTHONPATH"] = os.pathsep.join([project_root, existing_pythonpath])
    else:
        child_env["PYTHONPATH"] = project_root

    inputs_path: str | None = None
    try:
        if mlir_source:
            input_specs = parse_input_spec(mlir_source)
            if input_specs:
                inputs = _generate_deterministic_inputs(input_specs, seed=input_seed)
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".json", delete=False, encoding="utf-8"
                ) as inputs_file:
                    json.dump(_serialize_inputs(inputs), inputs_file)
                    inputs_path = inputs_file.name

        with tempfile.NamedTemporaryFile(suffix=".vmfb") as module_file, tempfile.NamedTemporaryFile(
            mode="w+", suffix=".json"
        ) as result_file:
            module_file.write(compiled_module)
            module_file.flush()

            runner_code = (
                "from execution_sandbox_agent.tools.execution import _run_compiled_from_file; "
                "import json, sys; "
                "inputs_file = sys.argv[4] if len(sys.argv) > 4 else None; "
                "result = _run_compiled_from_file(sys.argv[1], sys.argv[2], inputs_file); "
                "open(sys.argv[3], 'w', encoding='utf-8').write(json.dumps(result))"
            )

            cmd = [sys.executable, "-c", runner_code, module_file.name, driver_name, result_file.name]
            if inputs_path:
                cmd.append(inputs_path)

            completed = subprocess.run(
                cmd,
                cwd=project_root,
                env=child_env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )

            result_file.seek(0)
            result_text = result_file.read().strip()
            if result_text:
                return json.loads(result_text)

            error_output = "\n".join(
                item for item in [completed.stderr.strip(), completed.stdout.strip()] if item
            )
            if completed.returncode == 0:
                return {
                    "status": "error",
                    "error_message": "Execution finished without returning a result",
                }
            return {
                "status": "error",
                "error_message": error_output or f"Execution process exited with code {completed.returncode}",
            }
    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "error_message": f"Execution exceeded {timeout_seconds} seconds",
        }
    except Exception as exc:
        return {"status": "error", "error_message": str(exc)}
    finally:
        if inputs_path:
            try:
                os.unlink(inputs_path)
            except Exception:
                pass


def compare_outputs(output_a: Any, output_b: Any) -> dict:
    """
    Compare two program outputs. Both inputs are raw Python values
    (lists, scalars, bools) from _normalize_result() — not strings.
    """
    array_a = _to_numpy_if_numeric(output_a)
    array_b = _to_numpy_if_numeric(output_b)

    if array_a is not None and array_b is not None:
        if array_a.shape != array_b.shape:
            return {
                "diverges": True,
                "details": f"Shapes differ: {array_a.shape} vs {array_b.shape}",
            }

        matches = np.array_equal(array_a, array_b)
        if matches:
            return {"diverges": False, "details": "Outputs match exactly."}

        # Compute element-wise delta for the record
        delta = (array_a.astype(float) - array_b.astype(float)).tolist()
        return {
            "diverges": True,
            "details": f"Outputs diverge. Delta: {delta}",
        }

    # Scalar / bool fallback
    diverges = output_a != output_b
    return {
        "diverges": diverges,
        "details": "Outputs match exactly." if not diverges else "Outputs diverge.",
    }


def generate_semantic_explanation(
    program_a: str,
    program_b: str,
    output_a: str,
    output_b: str,
    outputs_diverge: bool,
    dialect: str,
) -> str:
    """
    Use Gemini to generate a semantic explanation of why two MLIR programs
    produce different or identical outputs.
    """
    try:
        dotenv_module = importlib.import_module("dotenv")
        messages_module = importlib.import_module("langchain_core.messages")
        genai_module = importlib.import_module("langchain_google_genai")

        load_dotenv = getattr(dotenv_module, "load_dotenv")
        HumanMessage = getattr(messages_module, "HumanMessage")
        SystemMessage = getattr(messages_module, "SystemMessage")
        ChatGoogleGenerativeAI = getattr(genai_module, "ChatGoogleGenerativeAI")

        load_dotenv()

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        response = llm.invoke([
            SystemMessage(
                content=(
                    "You are a compiler semantics analyst for the MASCoD benchmark.\n\n"
                    "You are given:\n"
                    "- Two MLIR programs (Program A = original, Program B = mutated)\n"
                    "- The mutation tool that was applied\n"
                    "- The computed outputs of both programs\n"
                    "- Whether the outputs diverge\n\n"
                    "Your job: Explain in 2-3 sentences WHY the mutation did or did not change the program's behavior. "
                    "Focus on the semantic effect of the code change.\n\n"
                    "Be precise and technical. Reference specific operations, values, or properties "
                    "(e.g., commutativity, operator semantics, constant propagation). Do not repeat "
                    "the outputs verbatim - explain the underlying reason.\n\n"
                    "If the outputs do not diverge, explain why the mutation was semantically neutral.\n"
                    "If the outputs diverge, explain what the mutation changed about the computation."
                )
            ),
            HumanMessage(
                content=(
                    f"Dialect: {dialect}\n"
                    f"Program A (original):\n{program_a}\n\n"
                    f"Program B (mutated):\n{program_b}\n\n"
                    f"Output A: {output_a}\n"
                    f"Output B: {output_b}\n"
                    f"Outputs diverge: {outputs_diverge}\n\n"
                    "Explain the semantic effect of this mutation."
                )
            ),
        ])

        content = response.content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("text"):
                    parts.append(item["text"])
                else:
                    parts.append(str(item))
            explanation = "".join(parts).strip()
        else:
            explanation = str(content).strip()

        if explanation:
            return explanation
        return "Semantic explanation unavailable: empty model response"
    except Exception as exc:
        return f"Semantic explanation unavailable: {exc}"


def compile_and_run_mlir(
    mlir_code: str,
    dialect: str = "linalg",
    target_backend: str = "llvm-cpu",
    timeout_seconds: int = 30,
    input_seed: int = 42,
) -> dict:
    """
    Compile and run an MLIR program in one step.

    Returns:
        dict with keys:
            - "compilation_status": str
            - "output": str | None
            - "error_message": str | None
    """
    compile_result = compile_mlir(
        mlir_code=mlir_code,
        dialect=dialect,
        target_backend=target_backend,
    )
    if compile_result["status"] != "success":
        error_message = compile_result.get("error_message", "Unknown compilation error")
        return {
            "compilation_status": f"error: {error_message}",
            "execution_status": "skipped",
            "output": None,
            "error_message": error_message,
        }

    run_result = run_compiled(
        compiled_module=compile_result["compiled_module"],
        mlir_source=mlir_code,
        target_backend=target_backend,
        timeout_seconds=timeout_seconds,
        input_seed=input_seed,
    )
    if run_result["status"] == "success":
        return {
            "compilation_status": "success",
            "execution_status": "success",
            "output": run_result.get("output"),
            "error_message": None,
            "compiled_module": compile_result["compiled_module"],
        }

    error_message = run_result.get("error_message", "Execution failed")
    return {
        "compilation_status": "success",
        "execution_status": run_result["status"],
        "output": None,
        "error_message": error_message,
        "compiled_module": compile_result["compiled_module"],
    }


EXECUTION_TOOLS = {
    "compile_mlir": compile_mlir,
    "run_compiled": run_compiled,
    "compare_outputs": compare_outputs,
    "compile_and_run_mlir": compile_and_run_mlir,
    "generate_semantic_explanation": generate_semantic_explanation,
    "parse_input_spec": parse_input_spec,
}
