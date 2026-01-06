import itertools
from typing import Dict, Tuple

class ClassicalQUBOSolver:
    """
    Exact brute-force solver for small QUBO problems.
    Used for benchmarking quantum annealing.
    """

    def solve(self, Q: Dict[Tuple[int, int], float], n_vars: int):
        best_energy = float("inf")
        best_solution = None

        for x in itertools.product([0, 1], repeat=n_vars):
            energy = 0.0
            for (i, j), v in Q.items():
                energy += v * x[i] * x[j]

            if energy < best_energy:
                best_energy = energy
                best_solution = x

        return best_solution, best_energy
