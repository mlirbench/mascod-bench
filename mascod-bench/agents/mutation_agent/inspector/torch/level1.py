import re

def torch_inspect_level1(input_mlir: str):
    # SSA variables
    variables = re.findall(r"%[\w\d\.]+", input_mlir)

    # Torch ops (torch.aten.*, torch.constant.*, etc.)
    ops = re.findall(r"(torch\.\w+(?:\.\w+)*)", input_mlir)

    # Dependency graph
    lines = input_mlir.split("\n")
    dependencies = {}

    for line in lines:
        if "=" in line:
            parts = line.split("=")
            if len(parts) >= 2:
                lhs = parts[0]
                rhs = "=".join(parts[1:])

                lhs_vars = re.findall(r"%[\w\d\.]+", lhs)
                rhs_vars = re.findall(r"%[\w\d\.]+", rhs)

                for var in lhs_vars:
                    dependencies[var] = rhs_vars

    return {
        "variables": list(set(variables)),
        "num_variables": len(set(variables)),
        "operations": list(set(ops)),
        "num_operations": len(ops),
        "dependency_graph": dependencies
    }