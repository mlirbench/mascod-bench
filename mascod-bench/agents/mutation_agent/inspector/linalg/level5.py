import re

def inspect_level5(input_mlir: str):
    variables = re.findall(r"%[\w\d]+", input_mlir)
    ops = re.findall(r"(\w+\.\w+)", input_mlir)

    shapes = re.findall(r"tensor<([^>]+)>", input_mlir)
    types = re.findall(r"(f\d+|i\d+)", input_mlir)

    return {
        "variables": list(set(variables)),
        "ops": list(set(ops)),
        "tensor_shapes": shapes,
        "element_types": types,
        "failure_points": {
            "can_break_ssa": variables,
            "can_break_types": types,
            "can_break_ops": ops
        }
    }