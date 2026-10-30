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
├── requirements.txt           # numpy, matplotlib, taichi, cupy-cuda12x
├── lif_pure_python.py         # Zero-dependency single-neuron LIF
├── lif_kernel.py              # LIFLayer class — CPU (numpy) + GPU (CuPy RawKernel)
├── lif_GPU_parallel.py        # LIFLayer class — Taichi/Vulkan backend
└── benchmark/
    ├── README.md              # Benchmark documentation
    └── run_benchmark.py       # Unified benchmark script (all backends)
```

## Setup

```bash
cd lif_kernel
pip install -r requirements.txt
```

## Usage

```bash
# Single-neuron reference
python lif_pure_python.py

# LIFLayer class with CPU + CuPy RawKernel (requires CUDA)
python lif_kernel.py

# Taichi/Vulkan backend (no CUDA needed, requires Python 3.12)
python lif_GPU_parallel.py

# Unified benchmarks (CPU always works, GPU/Taichi auto-detected)
python benchmark/run_benchmark.py
```

## Deliverables

- [x] `lif_pure_python.py` — zero-dependency LIF neuron
- [x] `lif_kernel.py` — LIFLayer with CPU + CuPy RawKernel backends
- [x] `lif_GPU_parallel.py` — LIFLayer with Taichi/Vulkan backend
- [x] `benchmark/run_benchmark.py` — unified benchmark with plots and CSV
