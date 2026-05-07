from pydantic import BaseModel
from typing import Literal, TypedDict, Optional, List, Dict, Any


class RootPayload(BaseModel):
    input_mlir: str
    dialect: Literal["linalg", "torch"]
    level: int
    # requirement_id: Optional[str] = None

class MASCoDState(TypedDict, total=False):
    # ---- User Input ----
    raw_input: str
    user_selected_dialect: str

    # ---- Generated Payload ----
    root_payload: Optional[Dict[str, Any]]

    # ---- Phase 2a Bridge Output ----
    phase2a_payload: Optional[Dict[str, Any]]  # {"programA": ..., "programB": ..., "dialect": ...}

    # ---- Phase 2b: Execution Sandbox Output ----
    # Includes execution outputs plus:
    # - semantic_explanation: str | None
    # - status: "success" | "compile_error" | "runtime_error" | "skipped"
    phase2b_payload: Optional[Dict[str, Any]]

    # ---- Logs ----
    logs: List[str]
