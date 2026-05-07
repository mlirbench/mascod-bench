import re

def torch_inspect_level5(input_mlir: str):
    variables = re.findall(r"%[\w\d\.]+", input_mlir)
    ops = re.findall(r"(torch\.\w+(?:\.\w+)*)", input_mlir)

    shapes = re.findall(r"!torch\.vtensor<([^>]+)>", input_mlir)
    dtypes = re.findall(r"!torch\.vtensor<[^,]+,([^>]+)>", input_mlir)

    return {
        "variables": list(set(variables)),
        "ops": list(set(ops)),
        "tensor_shapes": shapes,
        "element_types": dtypes,
        "failure_points": {
            "can_break_ssa": variables,
            "can_break_types": dtypes,
            "can_break_ops": ops
        }
    }