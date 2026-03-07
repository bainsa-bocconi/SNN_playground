# Task C — GPU Kernel Skeleton for LIF Neuron Simulation

**Assigned to:** Jacopo Gonzini & Niccolò Pagano  
**Sprint:** 1 (6–16 March 2026)

## Objective

Design and implement the skeleton of a GPU kernel for simulating Leaky Integrate-and-Fire (LIF) spiking neurons, which will become the computational core of the platform.

## Required Reading

1. [SpikingJelly — Implement CuPy Neuron tutorial](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/en/cupy_neuron.html)
2. [SpikingJelly — Triton backend and FlexSN tutorial](https://spikingjelly.readthedocs.io/zh-cn/latest/tutorials/en/triton_flexsn.html)
3. Fang, W. et al. (2023). *SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence.* Science Advances, 9(40), eadi1480.
4. [snnTorch source code](https://github.com/jeshraghian/snntorch/tree/master/snntorch) — LIF neurons as recursive PyTorch modules.
5. NVIDIA CUDA C Programming Guide, Chapters 1–3 (or Metal Shading Language Guide for macOS).

## LIF Neuron Equations

```
V[t+1] = β · V[t] + I[t] − Vth · s[t]
s[t+1] = 1  if V[t+1] >= Vth
          0  otherwise
```

Parameters: `β` (decay), `Vth` (threshold), `Vreset` (reset potential).

## Module Interface

```python
class LIFLayer:
    def __init__(self, n_neurons: int, beta: float, threshold: float, backend: str = 'cpu'):
        ...

    def forward(self, input_current) -> tuple[spikes, membrane]:
        ...

    def reset(self):
        ...
```

## Exercises

- **C.1** Minimal LIF neuron in pure Python. Implement the equations above. Validate with constant input current.
- **C.2** GPU implementation (choose one):
  - **Option A (CUDA):** C kernel updating N neurons in parallel, compiled with `nvcc`, wrapped with `ctypes`/`pybind11`.
  - **Option B (CuPy/Triton, recommended):** `CuPy RawKernel` or Triton `@triton.jit`.
  - **Option C (Metal/MPS, macOS):** PyTorch MPS backend or Metal compute shader.
- **C.3** Benchmark CPU vs GPU for N = 1,000 / 10,000 / 100,000 neurons over T = 1,000 timesteps. Produce a comparison plot.
- **C.4** Define the `LIFLayer` module interface (see above).

## Work Division

| Person | Tasks |
|--------|-------|
| Niccolò Pagano | Interface design, backend selection, CUDA/Metal implementation |
| Jacopo Gonzini | Pure Python implementation, CuPy/Triton version, benchmarking |

## Folder Structure

```
lif_kernel/
├── README.md
├── requirements.txt           # numpy, torch, cupy (optional), triton (optional)
├── lif_neuron.py              # Pure Python LIF implementation (C.1) + LIFLayer class (C.4)
├── lif_kernel.cu              # CUDA kernel (C.2 Option A) — optional
├── lif_kernel_cupy.py         # CuPy/Triton kernel (C.2 Option B) — optional
└── benchmark/
    ├── run_benchmark.py       # Benchmark script (C.3)
    └── benchmark_results.png  # Generated comparison plot
```

## Setup

```bash
cd lif_kernel
pip install -r requirements.txt
# Run pure Python demo
python lif_neuron.py
# Run benchmark
python benchmark/run_benchmark.py
```

## Deliverables

- [ ] `lif_neuron.py` — pure Python LIF implementation and `LIFLayer` class
- [ ] GPU kernel implementation (at least one of the three options)
- [ ] `benchmark/run_benchmark.py` and `benchmark/benchmark_results.png`
- [ ] This `README.md` with setup and usage instructions
