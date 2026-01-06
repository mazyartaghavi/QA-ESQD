import time
import torch

class RuntimeProfiler:
    def __init__(self):
        self.records = []

    def measure(self, label: str, func, *args, **kwargs):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        result = func(*args, **kwargs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end = time.time()

        self.records.append({
            "component": label,
            "time_sec": end - start
        })
        return result

    def summary(self):
        return self.records
