# SNN Playground — BAINSA Bocconi

**Bocconi AI and Neuroscience Student Association (BAINSA)**  
Spiking Neural Networks Research Project, Spring 2026

## Team

| Name | Role | Contact |
|------|------|---------|
| **Niccolò Pagano** | Project Lead | niccolo.pagano@studbocconi.it |
| Mara Andronache | Member | — |
| Marco Ventrella | Member | — |
| Jacopo Gonzini | Member | — |

## Project Overview

This repository hosts the BAINSA SNN Playground — a research platform for exploring **Spiking Neural Networks (SNNs)** across reinforcement learning, theoretical foundations, and high-performance GPU simulation.

The core reference reading for all members:

> Eshraghian, J.K. et al. (2023). *Training Spiking Neural Networks Using Lessons From Deep Learning.* Proceedings of the IEEE, 111(9), 1016–1054. [arXiv:2109.12894](https://arxiv.org/abs/2109.12894)

---

## Sprint 1 (6–16 March 2026)

### Task A — Reinforcement Learning Environment Library
**Assigned to:** Mara Andronache  
**Folder:** [`snn_rl_envs/`](snn_rl_envs/)

Set up a unified wrapper library exposing standardised APIs for RL environments (Gymnasium + PettingZoo), to be used as testbeds for SNN-based agents.

**Deliverables:**
- Working wrapper with `reset()`, `step()`, `get_action_space()`, `get_state_space()`, `render()` API
- Implementations: Tic-Tac-Toe (PettingZoo), CartPole (Gymnasium), Snake (custom)
- Demo script running a random agent on each environment
- `README.md` documenting available environments and usage

### Task B — Theoretical Results: ANN-to-SNN Reduction and Expressivity
**Assigned to:** Marco Ventrella  
**Folder:** [`theoretical_results/`](theoretical_results/)

Formally prove that any deep neural network can be reduced to an SNN, establish universal approximation, and derive associated bounds.

**Deliverables:**
- LaTeX document (PDF) with proofs for B.1–B.3 (ANN→SNN reduction, universal approximation, size bounds)
- One page of notes on heuristic SNN architectures (B.4, bonus)

### Task C — GPU Kernel Skeleton for LIF Neuron Simulation
**Assigned to:** Jacopo Gonzini & Niccolò Pagano  
**Folder:** [`lif_kernel/`](lif_kernel/)

Design and implement the skeleton of a GPU kernel for simulating Leaky Integrate-and-Fire (LIF) spiking neurons.

**Deliverables:**
- `lif_kernel.py` (or `.cu` + Python wrapper) with a working LIF neuron
- CPU vs GPU benchmark with a comparison plot
- `README.md` with setup and usage instructions

---

## Repository Structure

```
SNN_playground/
├── README.md                  # This file
├── snn_rl_envs/               # Task A — RL environment wrapper library
│   ├── README.md
│   ├── envs/                  # Concrete environment implementations
│   └── demo_random_agent.py   # Demo script
├── theoretical_results/       # Task B — Theoretical proofs and notes
│   ├── README.md
│   ├── proofs/                # LaTeX source files
│   └── notes/                 # Architecture notes
└── lif_kernel/                # Task C — LIF GPU kernel
    ├── README.md
    ├── lif_neuron.py          # Pure Python reference implementation
    └── benchmark/             # Benchmarking scripts and plots
```

---

## Guidelines

- **AI assistance:** Permitted for software development. _Not_ permitted for mathematical proofs and theoretical analysis (Task B).
- **Branches:** One branch per person; merge to `main` only after review.
- **Updates:** Group chat updates at least every 2 days.
- **Code style:** Follow PEP 8 for Python. LaTeX for all theoretical write-ups.
