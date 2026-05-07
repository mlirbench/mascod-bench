from .tools.execution import (
    compare_outputs,
    compile_and_run_mlir,
    generate_semantic_explanation,
)

from .tools.benchmark import benchmark_vmfb
from .tools.mlir_utils import parse_input_spec, make_input_flags
from .tools.mlir_embed import embed_inputs


def _describe_failure(label: str, result: dict) -> str | None:
    compilation_status = result.get("compilation_status", "error")
    execution_status = result.get("execution_status", "unknown")
    error_message = result.get("error_message") or "unknown error"

    if compilation_status != "success":
        return f"{label} failed to compile: {error_message}"

    if execution_status != "success":
        return f"{label} failed at runtime: {execution_status}: {error_message}"

    if result.get("output") is None:
        return f"{label} produced no output."

    return None


def _classify_status(result_a: dict, result_b: dict) -> str:
    if (
        result_a.get("compilation_status") != "success"
        or result_b.get("compilation_status") != "success"
    ):
        return "compile_error"

    if (
        result_a.get("execution_status") != "success"
        or result_b.get("execution_status") != "success"
        or result_a.get("output") is None
        or result_b.get("output") is None
    ):
        return "runtime_error"

    return "success"


def execution_sandbox_agent(state: dict) -> dict:
    """
    LangGraph node: Execution Sandbox Agent

    Reads phase2a_payload (programA, programB, dialect).
    Compiles and executes both programs using IREE.
    Compares outputs to produce ground truth.
    Writes phase2b_payload with results.
    """
    logs = state.get("logs", [])
    logs.append("Execution Sandbox Agent: Starting")

    payload = state.get(
        "toExecutionAgent"
    )

    if not payload:
        logs.append("Execution Sandbox Agent: Missing blind_results")
        return {
            **state,
            "phase2b_payload": {
                "programA": None,
                "programB": None,
                "programA_output": None,
                "programB_output": None,
                "compilation_status_A": "error: missing phase2a_payload",
                "compilation_status_B": "error: missing phase2a_payload",
                "outputs_diverge": None,
                "divergence_details": "Execution sandbox could not run because phase2a_payload is missing.",
                "semantic_explanation": None,
                "status": "skipped",
                "dialect": None,
                "tool_used": "unknown",
            },
            "logs": logs,
        }

    program_a = payload.get(
        "input_mlir"
    )

    program_b = payload.get(
        "mutated_mlir"
    )

    dialect = payload.get("dialect")

    if not program_a or not program_b:
        logs.append("Execution Sandbox Agent: blind_results is incomplete")
        return {
            **state,
            "phase2b_payload": {
                "programA": program_a,
                "programB": program_b,
                "programA_output": None,
                "programB_output": None,
                "compilation_status_A": "error: incomplete phase2a_payload",
                "compilation_status_B": "error: incomplete phase2a_payload",
                "outputs_diverge": None,
                "divergence_details": "Execution sandbox could not run because phase2a_payload is missing required fields.",
                "semantic_explanation": None,
                "status": "skipped",
                "dialect": dialect,
                "tool_used": None,
            },
            "logs": logs,
        }

    logs.append("Execution Sandbox Agent: Running Program A")
    result_a = compile_and_run_mlir(program_a, dialect=dialect)

    logs.append("Execution Sandbox Agent: Running Program B")
    result_b = compile_and_run_mlir(program_b, dialect=dialect)

    # ====================
    # BENCHMARK - PROGRAM A
    # ====================
    input_flags_program_a = make_input_flags(parse_input_spec(program_a))
    benchmark_a = benchmark_vmfb(result_a.get("compiled_module"), input_flags_program_a)

    # ====================
    # BENCHMARK - PROGRAM B
    # ====================
    input_flags_program_b = make_input_flags(parse_input_spec(program_b))
    benchmark_b = benchmark_vmfb(result_b.get("compiled_module"), input_flags_program_b)

    # ====================
    # EMBED_INPUTS - PROGRAM A
    # ====================
    self_containedA, embed_inputs_programA = embed_inputs(program_a)

    # ====================
    # EMBED_INPUTS - PROGRAM B
    # ====================
    self_containedB, embed_inputs_programB = embed_inputs(program_b)

    output_a = result_a.get("output")
    output_b = result_b.get("output")
    compilation_status_a = result_a.get("compilation_status", "error: unknown failure")
    compilation_status_b = result_b.get("compilation_status", "error: unknown failure")

    status = _classify_status(result_a, result_b)
    semantic_explanation = None

    if status == "success":
        logs.append("Execution Sandbox Agent: Comparing outputs")
        comparison = compare_outputs(output_a, output_b)
        outputs_diverge = comparison.get("diverges")
        divergence_details = comparison.get("details", "Outputs compared.")
        logs.append(f"Execution Sandbox Agent: Comparison complete, diverges={outputs_diverge}")

        semantic_explanation = generate_semantic_explanation(
            program_a=program_a,
            program_b=program_b,
            output_a=output_a,
            output_b=output_b,
            outputs_diverge=outputs_diverge,
            dialect=dialect,
        )
        logs.append("Execution Sandbox Agent: Semantic explanation generated")
    else:
        outputs_diverge = None
        failures = [
            _describe_failure("Program A", result_a),
            _describe_failure("Program B", result_b),
        ]
        divergence_details = " | ".join(item for item in failures if item)
        logs.append(f"Execution Sandbox Agent: {divergence_details}")

    logs.append("Execution Sandbox Agent: Finished")

    return {
        **state,
        "Agents": {
            **state.get("Agents", {}),
            "Execution_Agent": {
                "execution_results": {
                    "programA_output": output_a,
                    "programB_output": output_b,
                    "compilation_status_A": compilation_status_a,
                    "compilation_status_B": compilation_status_b,
                    "outputs_diverge": outputs_diverge,
                    "divergence_details": divergence_details,
                    "semantic_explanation": semantic_explanation,
                    "status": status,
                },
                "benchmark_results": {
                    "benchmark_result_program_a": benchmark_a,
                    "benchmark_result_program_b": benchmark_b
                },
                "embed_inputs": {
                    "embed_inputs_program_a": embed_inputs_programA,
                    "embed_inputs_program_b": embed_inputs_programB
                },
                "toLLMReasoning": {
                    "programA": program_a,
                    "programB": program_b,
                    "dialect": dialect,
                    "embed_inputs": {
                        "programA": embed_inputs_programA,
                        "programB": embed_inputs_programB,
                    }
                }
            }
        },
        "logs": logs
    }
