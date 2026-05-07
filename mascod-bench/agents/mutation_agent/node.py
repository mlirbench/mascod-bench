from agents.mutation_agent.selector import pick_requirement, get_requirement_by_id
from agents.mutation_agent.prompts.prompt_builder import build_user_prompt
from agents.mutation_agent.llm import generate_mutation
from services.mlir.validator_client import validate_mutation, validate_torch_mutation
from services.dataset.logger import save_run

from agents.mutation_agent.inspector.linalg.level1 import inspect_level1
from agents.mutation_agent.inspector.linalg.level2 import inspect_level2
from agents.mutation_agent.inspector.linalg.level3 import inspect_level3
from agents.mutation_agent.inspector.linalg.level4 import inspect_level4
from agents.mutation_agent.inspector.linalg.level5 import inspect_level5

from agents.mutation_agent.inspector.torch.level1 import torch_inspect_level1
from agents.mutation_agent.inspector.torch.level2 import torch_inspect_level2
from agents.mutation_agent.inspector.torch.level3 import torch_inspect_level3
from agents.mutation_agent.inspector.torch.level4 import torch_inspect_level4
from agents.mutation_agent.inspector.torch.level5 import torch_inspect_level5


LINALG_INSPECTORS = {
    1: inspect_level1,
    2: inspect_level2,
    3: inspect_level3,
    4: inspect_level4,
    5: inspect_level5
}

TORCH_INSPECTORS = {
    1: torch_inspect_level1,
    2: torch_inspect_level2,
    3: torch_inspect_level3,
    4: torch_inspect_level4,
    5: torch_inspect_level5
}


MAX_ATTEMPTS = 3


def mutation_agent(payload: dict):
    input_mlir = payload["input_mlir"]
    dialect = payload["dialect"]
    level = payload["level"]

    print(payload)

    # 1. Inspect
    if dialect == "linalg":
        metadata = LINALG_INSPECTORS[level](input_mlir)
    elif dialect == "torch":
        metadata = TORCH_INSPECTORS[level](input_mlir)

    # 2. Requirement
    requirement = pick_requirement(input_mlir, dialect, level)

    # 3. Prompt
    prompt = build_user_prompt(input_mlir, requirement, metadata)

    mutated_mlir = None
    validation_result = {"valid": False}
    attempts = 0
    attempt_history = []

    # 4. Retry loop
    for i in range(MAX_ATTEMPTS):
        attempts += 1

        try:
            candidate = generate_mutation(prompt)

            if dialect == "linalg":
                validation_result = validate_mutation(input_mlir, candidate)
            elif dialect == "torch":
                validation_result = validate_torch_mutation(input_mlir, candidate)

            attempt_entry = {
                "attempt": attempts,
                "candidate": candidate,
                "validation": validation_result
            }

            attempt_history.append(attempt_entry)

            if validation_result.get("valid"):
                mutated_mlir = candidate
                break

        except Exception as e:
            error_entry = {
                "attempt": attempts,
                "candidate": None,
                "validation": {"valid": False, "error": str(e)}
            }
            attempt_history.append(error_entry)

    # If still invalid
    if mutated_mlir is None:
        # mutated_mlir = attempt_history[-1]["candidate"] if attempt_history else "GENERATION_FAILED"
        mutated_mlir = (
            attempt_history[-1]["candidate"]
            if attempt_history and attempt_history[-1]["candidate"]
            else "GENERATION_FAILED"
        )

    # 5. Outputs

    metadata_output = {
        "level": level,
        "dialect": dialect,
        "requirement": requirement,
        "mlir-opt-verify-diagnostics": {
            "attempts": attempts,
            "final_validation": attempt_history[-1]["validation"] if attempt_history else {},
            "history": attempt_history
        },
        "input_mlir": input_mlir,
        "mutated_mlir": mutated_mlir,
        "metadata": metadata,
        # "validation": validation_result,
        # "attempts": attempts
    }

    blind_output = {
        "input_mlir": input_mlir,
        "mutated_mlir": mutated_mlir
    }

    toExecutionAgent = {
        "input_mlir": input_mlir,
        "mutated_mlir": mutated_mlir,
        "dialect" : dialect
    }

    # 6. Save dataset
    save_run({
        "metadata_output": metadata_output,
        "blind_output": blind_output
    })

    return {
        "Agents": {
            "Mutation_Agent": {
                "context_results": metadata_output,
                "blind_results": blind_output,
                "toExecutionAgent" : toExecutionAgent 
            }
        }
    }