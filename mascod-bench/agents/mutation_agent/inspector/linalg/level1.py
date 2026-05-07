import re

def inspect_level1(input_mlir: str):
    variables = re.findall(r"%[\w\d]+", input_mlir)
    ops = re.findall(r"(\w+\.\w+)", input_mlir)

    # Build simple dependency map
    lines = input_mlir.split("\n")
    dependencies = {}

    for line in lines:
        if "=" in line:
            lhs, rhs = line.split("=", 1)
            lhs_vars = re.findall(r"%[\w\d]+", lhs)
            rhs_vars = re.findall(r"%[\w\d]+", rhs)

            for var in lhs_vars:
                dependencies[var] = rhs_vars

    return {
        "variables": list(set(variables)),
        "num_variables": len(set(variables)),
        "operations": list(set(ops)),
        "num_operations": len(ops),
        "dependency_graph": dependencies
    }