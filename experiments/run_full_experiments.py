import yaml
import numpy as np
from evaluation.runtime_profiler import RuntimeProfiler
from evaluation.statistical_tests import significance_test
from experiments.ablation_runner import run_ablation
from utils.logger import ExperimentLogger
from utils.reproducibility import set_global_seed

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run_all():
    set_global_seed(42)
    config = load_config("configs/experiment.yaml")
    logger = ExperimentLogger("results")

    profiler = RuntimeProfiler()

    results = {
        "Q-EvoQD": [],
        "ES-QD": [],
        "ES": []
    }

    for seed in range(5):
        set_global_seed(seed)

        res_qevoqD = profiler.measure(
            "Q-EvoQD",
            run_ablation,
            config
        )
        res_es_qd = profiler.measure(
            "ES-QD",
            run_ablation,
            config,
            disable_qa=True
        )
        res_es = profiler.measure(
            "ES",
            run_ablation,
            config,
            disable_qd=True,
            disable_qa=True
        )

        results["Q-EvoQD"].append(res_qevoqD["reward"])
        results["ES-QD"].append(res_es_qd["reward"])
        results["ES"].append(res_es["reward"])

    stats = {
        "Q-EvoQD_vs_ES": significance_test(
            results["Q-EvoQD"], results["ES"]
        ),
        "Q-EvoQD_vs_ES-QD": significance_test(
            results["Q-EvoQD"], results["ES-QD"]
        ),
        "runtime": profiler.summary()
    }

    logger.log("performance", results)
    logger.log("statistics", stats)

if __name__ == "__main__":
    run_all()
