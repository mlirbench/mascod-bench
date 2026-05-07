import re

def torch_inspect_level2(input_mlir: str):
    # Shapes from vtensor
    shapes = re.findall(r"!torch\.vtensor<([^>]+)>", input_mlir)

    # Extract dtype from vtensor
    dtypes = re.findall(r"!torch\.vtensor<[^,]+,([^>]+)>", input_mlir)

    # Torch constants (scalar + tensor)
    constants = re.findall(
        r"(torch\.constant\.\w+\s+[^\s]+|torch\.vtensor\.literal\([^)]*\))",
        input_mlir
    )

    return {
        "shapes": list(set(shapes)),
        "dtype": list(set(dtypes)),
        "constants": constants,
        "num_constants": len(constants)
    }