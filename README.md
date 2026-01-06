# Q-EvoQD: Quantum Annealing–Enhanced Quality Diversity for MARL

This repository contains the official implementation of **Q-EvoQD**, a hybrid
quantum–classical framework integrating **Evolution Strategies (ES)**,
**Quality Diversity (QD)**, and **Quantum Annealing (QA)** for scalable
multi-agent reinforcement learning.

## 🔬 Research Motivation
Modern MARL systems suffer from deceptive reward landscapes and premature
convergence. Q-EvoQD introduces a principled quantum optimization layer that
refines elite policies via QUBO-based annealing, improving exploration,
robustness, and convergence.

## 🧠 Key Components
- Evolution Strategies for gradient-free policy optimization
- Quality-Diversity archives for behavioral exploration
- QUBO-based quantum annealing refinement
- Classical solver fallback for reproducibility
- Statistical significance & runtime profiling

## 📊 Reproducing Experiments
```bash
python experiments/run_full_experiments.py
python evaluation/plot_results.py

