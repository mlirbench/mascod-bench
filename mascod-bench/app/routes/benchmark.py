from fastapi import APIRouter
from app.schemas.request import BenchmarkRequest
from graph.pipeline import build_graph

router = APIRouter()

graph = build_graph()

@router.get("/")
def base():
    return {"status":"Localhost running at port 8000"}

@router.post("/run-benchmark")
def run_benchmark(req: BenchmarkRequest):
    state = {
        "user_input": req.dict(),
        "logs": []
    }

    result = graph.invoke(state)

    return {
        "Agents": result.get("Agents", {}),
        "logs": result.get("logs", []),
    }
