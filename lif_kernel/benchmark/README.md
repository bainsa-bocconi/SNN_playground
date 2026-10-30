# Benchmarks — LIFLayer backends

## Usage

```bash
cd lif_kernel

# Run all available backends with defaults
python benchmark/run_benchmark.py

# Specific backends and sizes
python benchmark/run_benchmark.py --backends cpu,gpu --N 1000,10000 --T 10000

# Custom output directory
python benchmark/run_benchmark.py --output-dir benchmark/ --repeats 10
```

## CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--backends` | cpu,gpu,taichi | Comma-separated (auto-detected) |
| `--N` | 1000,10000,100000 | Neuron counts |
| `--T` | 10000 | Timesteps |
| `--repeats` | 5 | Timed repetitions per config |
| `--warmup` | 2 | Warmup repetitions before timing |
| `--output-dir` | benchmark/ | Output directory for plots and CSV |

## Output

### Plots (7 PNGs, overwrites on each run)

| File | Content |
|------|---------|
| `LIFruntimeComparison.png` | `run()` time vs N, one line per backend |
| `LIFruntimeComparison_taichi.png` | `run()` CPU vs Taichi (subset) |
| `LIFruntimeCPU.png` | Pure Python single neuron time vs T |
| `forward_N_crossover.png` | `forward()` time vs N, one line per backend |
| `forward_T_comparison.png` | `forward()` time vs T (CPU vs GPU) |
| `LIFtensiongraph.png` | Voltage trace of a single LIF neuron |
| `gpu_T_comparison.png` | GPU dispatch: `forward()` per-step vs `run()` fused |

### CSV

`benchmark_results.csv` — raw timing data for all runs

## Model

All backends share the same discrete-time LIF update:

```
V[t+1] = β · V[t] + I[t]
s[t+1] = 1  if V[t+1] ≥ Vth, else 0
V[t+1] = V[t+1] − Vth  (soft reset)
```

Parameters: β = 0.9, Vth = 1.0, Vinit = 0.0.

## Backends

| Backend | File | Hardware |
|---------|------|----------|
| `cpu` | `lif_kernel.py` (numpy) | Any CPU |
| `gpu` | `lif_kernel.py` (CuPy RawKernel) | NVIDIA GPU + CUDA 12.x |
| `taichi` | `lif_GPU_parallel.py` (Taichi/Vulkan) | Any GPU with Vulkan support |
