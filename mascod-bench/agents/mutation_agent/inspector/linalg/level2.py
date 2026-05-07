import re

def inspect_level2(input_mlir: str):
    shapes = re.findall(r"tensor<([^>]+)>", input_mlir)
    types = re.findall(r"(f\d+|i\d+)", input_mlir)

    constants = re.findall(r"constant\s+([^\s:]+)", input_mlir)

    return {
        "shapes": list(set(shapes)),
        "dtype": list(set(types)),
        "constants": constants,
        "num_constants": len(constants)
    }