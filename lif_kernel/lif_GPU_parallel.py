import time
import taichi as ti

ti.init(arch=ti.vulkan)

pattern = [0.01, 0.02, 0.03, 0.0, 0.015, 0.025, 0.005, 0.02, 0.01, 0.03]
pattern = [x * 2 for x in pattern]

V_FIRE = 1.0
V_ZERO = 0.0
RATE = 0.98
T = 10000
N_MAX = 100000
REPEATS = 20

pat_len = len(pattern)
pattern_f = ti.field(dtype=ti.f32, shape=pat_len)
V_par_f = ti.field(dtype=ti.f32, shape=N_MAX)

for i, v in enumerate(pattern):
    pattern_f[i] = v

@ti.kernel
def run_parallel(N: int, T: int):
    for i in range(N):
        v = 0.0
        for t in range(T):
            inp = pattern_f[t % pat_len]
            v = RATE * v + inp
            if v >= V_FIRE:
                v = V_ZERO
        V_par_f[i] = v

def run_gpu(N: int, T: int):
    run_parallel(N, T)
    ti.sync()

population_sizes = list(range(5000, N_MAX + 1, 5000))

for N in population_sizes:
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        run_gpu(N, T)
        times.append(time.perf_counter() - start)

    mean_time = sum(times) / REPEATS
    print(f"N={N:>7} | mean GPU execution time: {mean_time*1000:7.2f} ms")
