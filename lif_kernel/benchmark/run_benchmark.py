import argparse
import sys
import time
from pathlib import Path

import numpy as np

SRC = str(Path(__file__).resolve().parent.parent)
if SRC not in sys.path:
    sys.path.insert(0, SRC)

BETA = 0.9
THRESHOLD = 1.0
PATTERN = [0.1, 0.2, 0.3, 0.0, 0.15, 0.25, 0.05, 0.2, 0.1, 0.3]


class _CPULIFLayer:
    def __init__(self, n_neurons, beta=BETA, threshold=THRESHOLD, backend='cpu'):
        self.n_neurons = n_neurons
        self.beta = beta
        self.threshold = threshold
        self.V = np.zeros(n_neurons, dtype=np.float32)
        self.spikes = np.zeros(n_neurons, dtype=np.int8)

    def forward(self, inp):
        self.V = self.beta * self.V + inp
        self.spikes = (self.V >= self.threshold).astype(np.int8)
        self.V -= self.threshold * self.spikes
        return self.spikes, self.V.copy()

    def run(self, T, pattern):
        counts = np.zeros(self.n_neurons, dtype=np.int32)
        for t in range(T):
            inp = float(pattern[t % len(pattern)])
            self.V = self.beta * self.V + inp
            spikes = (self.V >= self.threshold).astype(np.int8)
            self.V -= self.threshold * spikes
            counts += spikes
        return counts

    def reset(self):
        self.V[:] = 0.0
        self.spikes[:] = 0


BACKENDS = {'cpu': _CPULIFLayer}

try:
    import cupy
    from lif_kernel import LIFLayer as _GPULayer
    BACKENDS['gpu'] = _GPULayer
except ImportError:
    pass

try:
    from lif_GPU_parallel import LIFLayer as _TaichiLayer
    BACKENDS['taichi'] = _TaichiLayer
except (ImportError, RuntimeError):
    pass


def time_run(N, T, backend_cls, repeats=5, warmup=2):
    for _ in range(warmup):
        layer = backend_cls(N)
        layer.run(T, PATTERN)
    times = []
    for _ in range(repeats):
        layer = backend_cls(N)
        t0 = time.perf_counter()
        layer.run(T, PATTERN)
        times.append(time.perf_counter() - t0)
    return np.mean(times), np.std(times)


def time_forward(N, T, backend_cls, repeats=5, warmup=2):
    for _ in range(warmup):
        layer = backend_cls(N)
        for t in range(T):
            layer.forward(PATTERN[t % len(PATTERN)])
    times = []
    for _ in range(repeats):
        layer = backend_cls(N)
        t0 = time.perf_counter()
        for t in range(T):
            layer.forward(PATTERN[t % len(PATTERN)])
        times.append(time.perf_counter() - t0)
    return np.mean(times), np.std(times)


def time_single_neuron(T, repeats=100):
    from lif_pure_python import Neuron, run
    times = []
    for _ in range(repeats):
        n = Neuron()
        t0 = time.perf_counter()
        run(T, n, PATTERN)
        times.append(time.perf_counter() - t0)
    return np.mean(times), np.std(times)


def plot_run_vs_N(results, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for label, data in results.items():
        ns, means, stds = zip(*data)
        ax.errorbar(ns, means, yerr=stds, marker='o', label=label)
    ax.set_xlabel('Neurons (N)')
    ax.set_ylabel('Time (s)')
    ax.set_title('run() — fused T-step')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / 'LIFruntimeComparison.png')
    plt.close(fig)

    if 'taichi' in results:
        fig, ax = plt.subplots()
        for label in ('cpu', 'taichi'):
            if label not in results:
                continue
            ns, means, stds = zip(*results[label])
            ax.errorbar(ns, means, yerr=stds, marker='o', label=label)
        ax.set_xlabel('Neurons (N)')
        ax.set_ylabel('Time (s)')
        ax.set_title('run() — CPU vs Taichi')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(Path(output_dir) / 'LIFruntimeComparison_taichi.png')
        plt.close(fig)


def plot_forward_vs_N(results, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for label, data in results.items():
        ns, means, stds = zip(*data)
        ax.errorbar(ns, means, yerr=stds, marker='o', label=label)
    ax.set_xlabel('Neurons (N)')
    ax.set_ylabel('Time (s)')
    ax.set_title('forward() per-step')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / 'forward_N_crossover.png')
    plt.close(fig)


def plot_forward_vs_T(results, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for label, data in results.items():
        ts, means, stds = zip(*data)
        ax.errorbar(ts, means, yerr=stds, marker='o', label=label)
    ax.set_xlabel('Timesteps (T)')
    ax.set_ylabel('Time (s)')
    ax.set_title('forward() varying T')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / 'forward_T_comparison.png')
    plt.close(fig)


def plot_single_neuron(results, output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ts, means, stds = zip(*results)
    ax.errorbar(ts, np.array(means) * 1000, yerr=np.array(stds) * 1000, marker='o')
    ax.set_xlabel('Timesteps (T)')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Pure Python single neuron')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / 'LIFruntimeCPU.png')
    plt.close(fig)


def plot_tensiongraph(output_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from lif_pure_python import Neuron, run
    n = Neuron()
    T = 200
    voltages, spikes = run(T, n, PATTERN)
    fig, ax = plt.subplots()
    ax.plot(range(T), voltages)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Membrane potential')
    ax.set_title('LIF Neuron Dynamics')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / 'LIFtensiongraph.png')
    plt.close(fig)
    print( 'Voltage trace saved to LIFtensiongraph.png')


def plot_gpu_dispatch_comparison(output_dir, repeats=5, warmup=2):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from lif_kernel import LIFLayer

    T_range = [1000, 5000, 10000, 20000, 30000]
    N = 1000

    def time_forward(N, T):
        for _ in range(warmup):
            layer = LIFLayer(N, backend='gpu')
            for t in range(T):
                layer.forward(PATTERN[t % len(PATTERN)])
        times = []
        for _ in range(repeats):
            layer = LIFLayer(N, backend='gpu')
            t0 = time.perf_counter()
            for t in range(T):
                layer.forward(PATTERN[t % len(PATTERN)])
            times.append(time.perf_counter() - t0)
        return np.mean(times), np.std(times)

    def time_run(N, T):
        for _ in range(warmup):
            layer = LIFLayer(N, backend='gpu')
            layer.run(T, PATTERN)
        times = []
        for _ in range(repeats):
            layer = LIFLayer(N, backend='gpu')
            t0 = time.perf_counter()
            layer.run(T, PATTERN)
            times.append(time.perf_counter() - t0)
        return np.mean(times), np.std(times)

    fwd_pts = []
    run_pts = []
    for t in T_range:
        m_fwd, s_fwd = time_forward(N, t)
        m_run, s_run = time_run(N, t)
        fwd_pts.append((t, m_fwd, s_fwd))
        run_pts.append((t, m_run, s_run))
        print(f'    T={t:<6}  forward={m_fwd:.4f}s  run={m_run:.4f}s')

    fig, ax = plt.subplots()
    ts_f, m_f, s_f = zip(*fwd_pts)
    ts_r, m_r, s_r = zip(*run_pts)
    ax.errorbar(ts_f, m_f, yerr=s_f, marker='o', label='forward() per-step')
    ax.errorbar(ts_r, m_r, yerr=s_r, marker='s', label='run() fused')
    ax.set_xlabel('Timesteps (T)')
    ax.set_ylabel('Time (s)')
    ax.set_title('GPU dispatch: forward() vs run()')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(output_dir) / 'gpu_T_comparison.png')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='LIFLayer unified benchmarks')
    ap.add_argument('--backends', default=','.join(BACKENDS),
                    help=f'Comma-separated backends. Available: {", ".join(BACKENDS)}')
    ap.add_argument('--N', default='1000,10000,100000',
                    help='Comma-separated neuron counts')
    ap.add_argument('--T', type=int, default=10000, help='Number of timesteps')
    ap.add_argument('--repeats', type=int, default=5,
                    help='Timed repetitions per configuration')
    ap.add_argument('--warmup', type=int, default=2,
                    help='Warmup repetitions before timing')
    ap.add_argument('--output-dir', default=Path(__file__).parent,
                    help='Directory for plots and CSV')
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = [b.strip() for b in args.backends.split(',') if b.strip()]
    N_values = [int(n) for n in args.N.split(',')]
    T = args.T
    repeats = args.repeats
    warmup = args.warmup

    print(f'Backends: {selected}')
    print(f'N values: {N_values}')
    print(f'T: {T}')
    print(f'Repeats: {repeats}')
    print()

    run_results = {}
    forward_results = {}

    for name in selected:
        cls = BACKENDS.get(name)
        if cls is None:
            print(f'  [SKIP] backend "{name}" not available')
            continue
        print(f'--- {name.upper()} ---')

        print(f'  run() varying N...')
        pts = []
        for N in N_values:
            mean, std = time_run(N, T, cls, repeats, warmup)
            print(f'    N={N:<7}  {mean:.4f}s ± {std:.4f}s')
            pts.append((N, mean, std))
        run_results[name] = pts

        if name != 'cpu':
            print(f'  forward() varying N...')
            pts = []
            for N in N_values:
                mean, std = time_forward(N, T, cls, repeats, warmup)
                print(f'    N={N:<7}  {mean:.4f}s ± {std:.4f}s')
                pts.append((N, mean, std))
            forward_results[name] = pts
        print()

    print(f'--- SINGLE NEURON ---')
    print(f'  run() varying T...')
    T_single = [5000, 25000, 50000, 75000, 100000]
    single_results = []
    for t in T_single:
        mean, std = time_single_neuron(t)
        print(f'    T={t:<6}  {mean*1000:.2f}ms ± {std*1000:.2f}ms')
        single_results.append((t, mean, std))
    print()

    print('--- VOLTAGE TRACE ---')
    print(f'  Generating LIFtensiongraph.png...')
    plot_tensiongraph(output_dir)

    print('Generating plots...')
    if run_results:
        plot_run_vs_N(run_results, output_dir)
    if forward_results:
        plot_forward_vs_N(forward_results, output_dir)
    if 'cpu' in run_results and 'gpu' in run_results:
        T_range = [1000, 5000, 10000, 20000, 30000]
        print(f'  forward() varying T (CPU vs GPU, N=10000)...')
        fwd_T = {}
        for name in ('cpu', 'gpu'):
            if name not in BACKENDS:
                continue
            cls = BACKENDS[name]
            pts = []
            for t in T_range:
                mean, std = time_forward(10000, t, cls, repeats, warmup)
                pts.append((t, mean, std))
            fwd_T[name] = pts
        plot_forward_vs_T(fwd_T, output_dir)
    if single_results:
        plot_single_neuron(single_results, output_dir)

    if 'gpu' in BACKENDS:
        print(f'  GPU dispatch comparison (forward vs run, N=1000)...')
        plot_gpu_dispatch_comparison(output_dir, repeats, warmup)

    csv_path = output_dir / 'benchmark_results.csv'
    with open(csv_path, 'w') as f:
        f.write('benchmark,backend,N,T,mean_time_s,std_time_s\n')
        for name, pts in run_results.items():
            for N, mean, std in pts:
                f.write(f'run,{name},{N},{T},{mean:.6f},{std:.6f}\n')
        for name, pts in forward_results.items():
            for N, mean, std in pts:
                f.write(f'forward,{name},{N},{T},{mean:.6f},{std:.6f}\n')
        for t, mean, std in single_results:
            f.write(f'single_neuron,pure_python,1,{t},{mean:.6f},{std:.6f}\n')

    print(f'\nResults saved to {csv_path}')
    print('Done.')


if __name__ == '__main__':
    main()
