from pydantic import BaseModel

class RootResponse(BaseModel):
    input_mlir: str
    dialect: str
    level: int