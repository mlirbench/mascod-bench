"""
execution_agent/run.py — One-liner public API for the execution sandbox.

Usage:
    from execution_agent import run_sandbox
    result = run_sandbox(program_a, program_b, dialect="linalg")
"""
from __future__ import annotations

from typing import Literal

from execution_agent.tools.execution import (
    compile_and_run_mlir,
    compare_outputs,
    generate_semantic_explanation,
)


def run_sandbox(
    program_a: str,
    program_b: str,
    dialect: Literal["linalg", "torch"],
    tool_used: str = "unknown",
) -> dict:
    """
    Compile, execute, and compare an MLIR program pair.

    Args:
        program_a:  MLIR source for the original program.
        program_b:  MLIR source for the mutated program.
        dialect:    "linalg" or "torch".
        tool_used:  Name of the mutation tool that produced program_b.

    Returns:
        phase2b_payload dict with keys:
            programA, programB, programA_output, programB_output,
            compilation_status_A, compilation_status_B,
            outputs_diverge, divergence_details, semantic_explanation,
            status, dialect, tool_used
    """
    result_a = compile_and_run_mlir(program_a, dialect)
    result_b = compile_and_run_mlir(program_b, dialect)

    comp_a = result_a.get("compilation_status", "error")
    comp_b = result_b.get("compilation_status", "error")
    exec_a = result_a.get("execution_status", "unknown")
    exec_b = result_b.get("execution_status", "unknown")

    a_ok = comp_a == "success" and exec_a == "success" and result_a.get("output") is not None
    b_ok = comp_b == "success" and exec_b == "success" and result_b.get("output") is not None

    if not a_ok or not b_ok:
        failures = []
        if not a_ok:
            msg = result_a.get("error_message") or f"compile={comp_a} exec={exec_a}"
            failures.append(f"A: {msg}")
        if not b_ok:
            msg = result_b.get("error_message") or f"compile={comp_b} exec={exec_b}"
            failures.append(f"B: {msg}")

        status = "compile_error" if (comp_a != "success" or comp_b != "success") else "runtime_error"

        return {
            "programA": program_a,
            "programB": program_b,
            "programA_output": result_a.get("output"),
            "programB_output": result_b.get("output"),
            "compilation_status_A": comp_a,
            "compilation_status_B": comp_b,
            "outputs_diverge": None,
            "divergence_details": " | ".join(failures),
            "semantic_explanation": None,
            "status": status,
            "dialect": dialect,
            "tool_used": tool_used,
        }

    output_a = result_a["output"]
    output_b = result_b["output"]
    comparison = compare_outputs(output_a, output_b)

    semantic_explanation = generate_semantic_explanation(
        program_a=program_a,
        program_b=program_b,
        output_a=output_a,
        output_b=output_b,
        outputs_diverge=comparison["diverges"],
        dialect=dialect,
        tool_used=tool_used,
    )

    return {
        "programA": program_a,
        "programB": program_b,
        "programA_output": output_a,
        "programB_output": output_b,
        "compilation_status_A": comp_a,
        "compilation_status_B": comp_b,
        "outputs_diverge": comparison["diverges"],
        "divergence_details": comparison["details"],
        "semantic_explanation": semantic_explanation,
        "status": "success",
        "dialect": dialect,
        "tool_used": tool_used,
    }
