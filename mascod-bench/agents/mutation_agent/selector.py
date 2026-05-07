import json
import hashlib
from pathlib import Path


BASE_PATH = Path(__file__).parent / "requirements"


def load_requirements(dialect: str):
    path = BASE_PATH / f"{dialect}.json"
    with open(path) as f:
        return json.load(f)


# def pick_requirement(input_mlir: str, dialect: str, level: int):
#     data = load_requirements(dialect)
#     key = f"level_{level}"

#     if key not in data or len(data[key]) == 0:
#         raise ValueError(f"No requirements for {dialect} level {level}")

#     requirements = data[key]

#     h = int(hashlib.sha256(input_mlir.encode()).hexdigest(), 16)
#     idx = h % len(requirements)

#     return requirements[idx]

def pick_requirement(input_mlir: str, dialect: str, level: int):
    data = load_requirements(dialect)
    key = f"level_{level}"

    requirements = data[key]

    idx = (len(input_mlir) + level) % len(requirements)
    return requirements[idx]

def get_requirement_by_id(dialect: str, level: int, requirement_id: str):
    data = load_requirements(dialect)
    key = f"level_{level}"

    for req in data.get(key, []):
        if req["id"] == requirement_id:
            return req

    # raise ValueError(f"Requirement {requirement_id} not found")
    if requirement_id is None:
        return None

    raise ValueError(f"Requirement {requirement_id} not found")