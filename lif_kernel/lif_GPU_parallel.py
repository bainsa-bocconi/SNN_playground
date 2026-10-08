import taichi as ti
import numpy as np

ti.init(arch=ti.vulkan)

BETA = 0.9
V_TH = 1.0


@ti.kernel
def _forward_kernel(V: ti.template(), spikes: ti.template(),
                    beta: ti.f32, threshold: ti.f32, inp: ti.f32):
    for i in range(V.shape[0]):
        v = beta * V[i] + inp
        if v >= threshold:
            spikes[i] = 1
            v -= threshold
        else:
            spikes[i] = 0
        V[i] = v


@ti.kernel
def _run_kernel(V: ti.template(), counts: ti.template(),
                beta: ti.f32, threshold: ti.f32,
                pattern: ti.types.ndarray(),
                T: ti.i32, pat_len: ti.i32):
    for i in range(V.shape[0]):
        v = 0.0
        cnt = 0
        for t in range(T):
            inp = pattern[t % pat_len]
            v = beta * v + inp
            if v >= threshold:
                cnt += 1
                v -= threshold
        V[i] = v
        counts[i] = cnt


class LIFLayer:
    def __init__(self, n_neurons, beta=BETA, threshold=V_TH, backend='taichi'):
        self.n_neurons = n_neurons
        self.beta = beta
        self.threshold = threshold
        self.backend = backend

        self.V = ti.field(dtype=ti.f32, shape=n_neurons)
        self.spikes = ti.field(dtype=ti.i32, shape=n_neurons)
        self.counts = ti.field(dtype=ti.i32, shape=n_neurons)

    def forward(self, input_current: float):
        _forward_kernel(self.V, self.spikes,
                        self.beta, self.threshold,
                        float(input_current))
        ti.sync()
        return self.spikes.to_numpy(), self.V.to_numpy()

    def run(self, T, pattern):
        self.reset()
        pat = np.array(pattern, dtype=np.float32)
        _run_kernel(self.V, self.counts,
                    self.beta, self.threshold,
                    pat, T, len(pattern))
        ti.sync()
        return self.counts.to_numpy()

    def reset(self):
        self.V.fill(0)
        self.spikes.fill(0)
        self.counts.fill(0)
