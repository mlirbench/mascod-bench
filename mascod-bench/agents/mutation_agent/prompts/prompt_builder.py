def build_user_prompt(input_mlir: str, requirement: dict, metadata: dict):
    return f"""
INPUT_MLIR_START
{input_mlir}
INPUT_MLIR_END

REQUIREMENT:
{requirement['description']}

RULES:
{chr(10).join("- " + r for r in requirement['rules'])}

METADATA:
{metadata}

TASK:
Apply the mutation requirement to the input MLIR program.

OUTPUT:
Return ONLY the mutated MLIR program.
"""