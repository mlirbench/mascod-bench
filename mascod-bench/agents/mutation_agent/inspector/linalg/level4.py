import re

def inspect_level4(input_mlir: str):
    blocks = input_mlir.count("{")
    regions = input_mlir.count("}")

    ops = re.findall(r"(\w+\.\w+)", input_mlir)

    # Build adjacency graph
    lines = input_mlir.split("\n")
    graph = []

    for line in lines:
        if "=" in line:
            lhs, rhs = line.split("=", 1)
            lhs_vars = re.findall(r"%[\w\d]+", lhs)
            rhs_vars = re.findall(r"%[\w\d]+", rhs)

            for var in lhs_vars:
                graph.append({
                    "node": var,
                    "depends_on": rhs_vars
                })

    return {
        "num_blocks": blocks,
        "num_regions": regions,
        "ops": list(set(ops)),
        "computation_graph": graph
    }