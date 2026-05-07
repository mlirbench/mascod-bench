from fastapi import FastAPI
from app.routes.benchmark import router as benchmark_router

app = FastAPI()

app.include_router(benchmark_router)