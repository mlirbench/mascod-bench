from pydantic import BaseModel
from typing import Optional, Literal

class BenchmarkRequest(BaseModel):
    input_mlir: str
    pair_id: Optional[str] = None
    dialect: Literal["linalg", "torch"]
    level: int
    # requirement_id: Optional[str] = None
