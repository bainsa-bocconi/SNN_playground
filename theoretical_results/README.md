# Task B — Theoretical Results: ANN-to-SNN Reduction and Expressivity

**Assigned to:** Marco Ventrella  
**Sprint:** 1 (6–16 March 2026)

> **Note:** AI assistance is **not** permitted for mathematical proofs and theoretical analysis in this task.

## Objective

Formally prove that any deep neural network can be reduced to a spiking neural network, establish that SNNs are universal function approximators, and derive associated bounds.

## Required Reading (in order)

1. Maass, W. (1997). *Networks of spiking neurons: The third generation of neural network models.* Neural Networks, 10(9), 1659–1671.  
   [PDF](https://igi-web.tugraz.at/people/maass/psfiles/85a.pdf)

2. Maass, W. (1996). *Lower bounds for the computational power of networks of spiking neurons.* Neural Computation, 8(1), 1–40.

3. Rueckauer, B. et al. (2017). *Conversion of Continuous-Valued Deep Networks to Efficient Event-Driven Networks for Image Classification.* Frontiers in Neuroscience, 11, 682.

4. Sengupta, A. et al. (2019). *Going Deeper in Spiking Neural Networks: VGG and Residual Architectures.* Frontiers in Neuroscience, 13, 95.

5. Neftci, E.O. et al. (2019). *Surrogate Gradient Learning in Spiking Neural Networks.* IEEE Signal Processing Magazine, 36(6), 51–63.

## Exercises

- **B.1** ANN→SNN Reduction Theorem. Formalize and prove: given a feedforward ReLU network `f: Rⁿ → Rᵐ` with `L` layers, there exists an SNN with LIF neurons approximating `f` with error `ε` in `T = O(1/ε)` timesteps.
- **B.2** Universal Approximation for SNNs. Show that SNNs with LIF neurons can approximate any continuous function `f: [0,1]ⁿ → R`.
- **B.3** Upper bounds on network size. Given an ANN with `N` neurons and `W` weights, show the equivalent SNN requires at most `O(N)` spiking neurons and `O(W)` synapses.
- **B.4 (Bonus)** Heuristic architectures — propose 2–3 SNN architecture ideas for a control task (CartPole/Pendulum) and a board-game task (Tic-Tac-Toe).

## Folder Structure

```
theoretical_results/
├── README.md
├── proofs/
│   ├── main.tex               # LaTeX source — proofs B.1–B.3
│   └── references.bib         # BibTeX bibliography
└── notes/
    └── architecture_notes.md  # Heuristic architecture ideas (B.4)
```

## Deliverables

- [ ] `proofs/main.tex` — LaTeX document with proofs for B.1–B.3 (draft/sketch level is acceptable)
- [ ] Compiled PDF
- [ ] `notes/architecture_notes.md` — one page of heuristic architecture ideas (B.4)
