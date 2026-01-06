import numpy as np
from typing import Dict, Tuple

class PolicyQUBOMapper:
    """
    Maps elite policy parameters to a QUBO formulation
    for quantum annealing-based refinement.
    """

    def __init__(self, lambda_reg: float = 0.1):
        self.lambda_reg = lambda_reg

    def discretize(self, weights: np.ndarray, bits: int = 4) -> np.ndarray:
        """
        Binary encoding of continuous parameters.
        """
        scale = 2 ** bits - 1
        return np.round((weights - weights.min()) /
                        (weights.max() - weights.min()) * scale).astype(int)

    def build_qubo(self, weights: np.ndarray) -> Dict[Tuple[int, int], float]:
        """
        Constructs QUBO matrix Q such that x^T Q x approximates
        policy loss + diversity regularization.
        """
        binary_weights = self.discretize(weights)
        Q = {}

        for i in range(len(binary_weights)):
            Q[(i, i)] = float(binary_weights[i] ** 2)

        for i in range(len(binary_weights)):
            for j in range(i + 1, len(binary_weights)):
                Q[(i, j)] = self.lambda_reg * binary_weights[i] * binary_weights[j]

        return Q
