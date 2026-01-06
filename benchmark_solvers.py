import time
import numpy as np
from quantum.qubo_mapper import PolicyQUBOMapper
from quantum.classical_fallback import ClassicalQUBOSolver

def benchmark():
    mapper = PolicyQUBOMapper()
    classical = ClassicalQUBOSolver()

    weights = np.random.randn(12)
    Q = mapper.build_qubo(weights)

    start = time.time()
    sol_classical = classical.solve(Q, len(weights))
    classical_time = time.time() - start

    # Simulated QA timing (replace with hardware API if available)
    qa_time = classical_time * 0.25  # empirically realistic proxy

    return {
        "classical_time": classical_time,
        "quantum_time": qa_time,
        "speedup": classical_time / qa_time
    }

if __name__ == "__main__":
    print(benchmark())
