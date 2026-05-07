import re

def inspect_level3(input_mlir: str):
    ops = re.findall(r"(\w+\.\w+)", input_mlir)

    op_signatures = []
    lines = input_mlir.split("\n")

    for line in lines:
        match = re.search(r"(\w+\.\w+)\((.*?)\)", line)
        if match:
            op_name = match.group(1)
            operands = re.findall(r"%[\w\d]+", match.group(2))

            op_signatures.append({
                "op": op_name,
                "operands": operands
            })

    return {
        "valid_ops": list(set(ops)),
        "op_signatures": op_signatures
    }