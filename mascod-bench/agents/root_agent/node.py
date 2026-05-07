from agents.root_agent.schema import RootPayload


SUPPORTED_DIALECTS = ["linalg", "torch"]
LEVELS = [1, 2, 3, 4, 5]


def root_agent(user_input: dict) -> RootPayload:
    input_mlir = user_input["input_mlir"]
    dialect = user_input["dialect"]
    level = user_input["level"]

    if dialect not in SUPPORTED_DIALECTS:
        raise ValueError("Invalid dialect")

    if level not in LEVELS:
        raise ValueError("Invalid level")

    return RootPayload(
        input_mlir=input_mlir,
        dialect=dialect,
        level=level,
        # requirement_id=user_input.get("requirement_id"),
    )
