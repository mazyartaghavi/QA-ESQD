import json
from pathlib import Path

class ExperimentLogger:
    def __init__(self, log_dir="logs"):
        self.dir = Path(log_dir)
        self.dir.mkdir(exist_ok=True)

    def log(self, name, data):
        with open(self.dir / f"{name}.json", "w") as f:
            json.dump(data, f, indent=2)
