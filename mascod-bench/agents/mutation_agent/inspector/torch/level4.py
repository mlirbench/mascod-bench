import re

def torch_inspect_level4(input_mlir: str):
    # Better block detection
    blocks = len(re.findall(r"func\.func|\^bb\d+", input_mlir))
    regions = input_mlir.count("{")

    ops = re.findall(r"(torch\.\w+(?:\.\w+)*)", input_mlir)

    # Computation graph
    lines = input_mlir.split("\n")
    graph = []

    for line in lines:
        if "=" in line:
            parts = line.split("=")
            if len(parts) >= 2:
                lhs = parts[0]
                rhs = "=".join(parts[1:])

                lhs_vars = re.findall(r"%[\w\d\.]+", lhs)
                rhs_vars = re.findall(r"%[\w\d\.]+", rhs)

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