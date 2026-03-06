# Task A — Reinforcement Learning Environment Library

**Assigned to:** Mara Andronache  
**Sprint:** 1 (6–16 March 2026)

## Objective

Build a unified wrapper library that exposes a standardised API for reinforcement learning environments (Gymnasium + PettingZoo), to serve as testbeds for SNN-based agents.

## Required Reading

1. [Gymnasium documentation](https://gymnasium.farama.org/) — focus on Classic Control and Toy Text environments  
   - Tutorial: [Create a Custom Environment](https://gymnasium.farama.org/introduction/create_custom_env/)
2. [PettingZoo documentation](https://pettingzoo.farama.org/) — focus on Classic environments (Tic-Tac-Toe, Connect4)
3. Towers, M. et al. (2024). *Gymnasium: A Standard Interface for Reinforcement Learning Environments.* [arXiv:2407.17032](https://arxiv.org/abs/2407.17032)

## Unified Wrapper API

All environments must implement the following interface:

```python
class BaseEnv:
    def reset(self) -> state: ...
    def step(self, action) -> (state, reward, done, info): ...
    def get_action_space(self) -> (size, type): ...
    def get_state_space(self) -> (size, type): ...
    def render(self): ...          # optional
```

## Environments

| Environment | Backend | Status |
|------------|---------|--------|
| Tic-Tac-Toe | PettingZoo `tictactoe_v3` | — |
| CartPole | Gymnasium `CartPole-v1` | — |
| Snake | Custom | — |

## Folder Structure

```
snn_rl_envs/
├── README.md
├── requirements.txt           # gymnasium, pettingzoo, numpy, ...
├── base_env.py                # Abstract base class / unified wrapper interface
├── envs/
│   ├── __init__.py
│   ├── tictactoe_env.py       # Tic-Tac-Toe wrapper
│   ├── cartpole_env.py        # CartPole wrapper
│   └── snake_env.py           # Custom Snake environment
└── demo_random_agent.py       # Runs a random agent on each environment
```

## Exercises

- **A.1** Install and test Gymnasium + PettingZoo. Run at least 3–4 different environments with random agents. Document setup instructions.
- **A.2** Design the unified wrapper interface (see API above).
- **A.3** Implement at least 3 concrete environments: Tic-Tac-Toe, CartPole, Snake (configurable grid size, wall wrapping toggle, optional obstacles).
- **A.4** Propose 3–5 additional environments suitable for SNN testing (discrete vs. continuous, partial vs. full observability, short vs. long time horizons).

## Setup

```bash
cd snn_rl_envs
pip install -r requirements.txt
python demo_random_agent.py
```

## Deliverables

- [ ] `base_env.py` — unified wrapper interface
- [ ] At least 3 environment implementations in `envs/`
- [ ] `demo_random_agent.py` running a random agent on each environment
- [ ] `requirements.txt`
- [ ] This `README.md` updated with usage instructions and environment descriptions
