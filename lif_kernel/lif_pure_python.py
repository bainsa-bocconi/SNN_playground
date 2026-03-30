import time
import matplotlib.pyplot as plt

pattern = [0.02, 0.04, 0.06, 0.0, 0.035, 0.035, 0.01, 0.02, 0.05, 0.1]

class Neuron:
    def __init__(self, V_fire=1.0, V_zero=0.0, rate=0.98, initialtension=0.0):
        self.V = initialtension
        self.V_fire = V_fire
        self.V_zero = V_zero
        self.rate = rate

    def fire(self):
        self.V = self.V_zero
        return 1

    def step(self, inp):
        self.V = self.rate * self.V + inp
        if self.V >= self.V_fire:
            spike = self.fire()
        else:
            spike = 0
        return self.V, spike


def run(T, lif: Neuron, inputlist=None):
    if inputlist is None:
        inputlist = [0.2]

    voltages = []
    spikes = []

    for i in range(T):
        inp = inputlist[i % len(inputlist)]
        v, s = lif.step(inp)
        voltages.append(v)
        spikes.append(s)

    return voltages, spikes

def computetime(V_fire=1.0, V_zero=0.0, rate=0.98, initialtension=0.0):
    meantime = []
    for n in range(5000, 100000, 5000):
        times = []
        for _ in range(100):
            lif = Neuron(V_fire, V_zero, rate, initialtension)
            start = time.perf_counter()
            voltages, spikes = run(n, lif, pattern)
            end = time.perf_counter()
            times.append(end - start)
        meantime.append(sum(times) / len(times))
        print(f"mean for {n} trials:", meantime[-1])
    return meantime, voltages, spikes

meantime, voltages, spikes = computetime()
print(f"execution meantime: {meantime}")
