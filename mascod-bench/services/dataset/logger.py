import json
import time
from pathlib import Path

BASE = Path("data/runs")


def save_run(data: dict):
    BASE.mkdir(parents=True, exist_ok=True)

    filename = BASE / f"run_{int(time.time()*1000)}.json"

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)