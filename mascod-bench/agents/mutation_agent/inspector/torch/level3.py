import re

def torch_inspect_level3(input_mlir: str):
    ops = re.findall(r"(torch\.\w+(?:\.\w+)*)", input_mlir)

    op_signatures = []
    lines = input_mlir.split("\n")

    for line in lines:
        op_match = re.search(r"(torch\.\w+(?:\.\w+)*)", line)
        if op_match:
            op_name = op_match.group(1)

            operands = re.findall(r"%[\w\d\.]+", line)

            op_signatures.append({
                "op": op_name,
                "operands": operands
            })

    return {
        "valid_ops": list(set(ops)),
        "op_signatures": op_signatures
    }