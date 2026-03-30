import numpy as np
import cupy as cp
import time

lif_step_kernel = cp.RawKernel(r'''
extern "C" __global__
void lif_step(float* V, int* spikes, const float inp,
              float beta, float threshold, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = beta * V[i] + inp;
    if (v >= threshold) {
        spikes[i] = 1;
        V[i] = v - threshold;
    } else {
        spikes[i] = 0;
        V[i] = v;
    }
}
''', 'lif_step')

lif_run_kernel = cp.RawKernel(r'''
extern "C" __global__
void lif_run(float* V, int* spike_counts, const float* pattern,
             float beta, float threshold, int T, int pat_len, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = 0.0f;
    for (int t = 0; t < T; t++) {
        float inp = pattern[t % pat_len];
        v = beta * v + inp;
        if (v >= threshold) {
            spike_counts[i]++;
            v = v - threshold;
        }
    }
    V[i] = v;
}
''', 'lif_run')


class LIFLayer:
    def __init__(self, n_neurons, beta=0.9, threshold=1.0, backend='cpu'):
        self.n_neurons = n_neurons
        self.beta      = beta
        self.threshold = threshold
        self.backend   = backend
        self.threads   = 256
        self.blocks    = (n_neurons + self.threads - 1) // self.threads

        if backend == 'cpu':
            self.V      = np.zeros(n_neurons, dtype=np.float32)
            self.spikes = np.zeros(n_neurons, dtype=np.int8)

        elif backend == 'gpu':
            self.V      = cp.zeros(n_neurons, dtype=cp.float32)
            self.spikes = cp.zeros(n_neurons, dtype=cp.int32)
            self.counts = cp.zeros(n_neurons, dtype=cp.int32)

    def forward(self, input_current: float):
        """
        One timestep. there are 2 backends, CPU and GPU.
        Returns spike array (N,) and membrane voltage (N,).
        Note: GPU forward() pays full dispatch overhead per call.
        Use run() for large T to avoid this.
        """
        if self.backend == 'cpu':
            self.V      = self.beta * self.V + input_current
            self.spikes = (self.V >= self.threshold).astype(np.int8)
            self.V     -= self.threshold * self.spikes
            return self.spikes, self.V.copy()

        elif self.backend == 'gpu':
            lif_step_kernel(
                (self.blocks,), (self.threads,),
                (self.V, self.spikes, np.float32(input_current),
                 self.beta, self.threshold, self.n_neurons)
            )
            cp.cuda.Stream.null.synchronize()
            return self.spikes, self.V

    def run(self, T, pattern):
        """
        T timesteps, cycling through pattern.
        Returns spike counts per neuron (N,) — same type for both backends.
        GPU: single kernel dispatch, T-loop fused inside.
        CPU: T-loop in Python, forward() called per step.
        """
        if self.backend == 'cpu':
            counts = np.zeros(self.n_neurons, dtype=np.int32)
            for t in range(T):
                inp         = float(pattern[t % len(pattern)])
                self.V      = self.beta * self.V + inp
                self.spikes = (self.V >= self.threshold).astype(np.int8)
                self.V     -= self.threshold * self.spikes
                counts     += self.spikes
            return counts

        elif self.backend == 'gpu':
            self.counts[:] = 0
            self.V[:]      = 0.0
            pat_gpu = cp.array(pattern, dtype=cp.float32)
            lif_run_kernel(
                (self.blocks,), (self.threads,),
                (self.V, self.counts, pat_gpu,
                 self.beta, self.threshold,
                 T, len(pattern), self.n_neurons)
            )
            cp.cuda.Stream.null.synchronize()
            return self.counts

    def reset(self):
        if self.backend == 'cpu':
            self.V[:]      = 0.0
            self.spikes[:] = 0
        elif self.backend == 'gpu':
            self.V[:]      = 0.0
            self.spikes[:] = 0
            self.counts[:] = 0

# ---- simulation zone ----

pattern = cp.array([0.1, 0.2, 0.3, 0.0, 0.15, 0.25, 0.05, 0.2, 0.1, 0.3], dtype=cp.float32)

def benchmark(N, T, backend='cpu', repeats=5, warmup=1):
    for _ in range(warmup):
        m = LIFLayer(N, backend=backend)
        for t in range(T):
            m.forward(float(pattern[t % len(pattern)]))

    times = []
    for _ in range(repeats):
        m = LIFLayer(N, backend=backend)
        start = time.perf_counter()
        for t in range(T):
            m.forward(float(pattern[t % len(pattern)]))
        end = time.perf_counter()
        times.append(end - start)

    times = np.array(times)
    return {"mean": times.mean(), "std": times.std(),
            "min": times.min(),  "max": times.max()}

N_values = [1000, 10000, 100000]
T = 10000

for backend in ("cpu", "gpu"):
    print(f"--- {backend.upper()} ---")
    for N in N_values:
        res = benchmark(N, T, backend=backend)
        print(f"  N={N:<7}  mean={res['mean']:.4f}s  "
              f"std={res['std']:.4f}s  "
              f"min={res['min']:.4f}s  max={res['max']:.4f}s")
    print()
