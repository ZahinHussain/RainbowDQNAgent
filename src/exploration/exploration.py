import numpy as np

class EpsilonScheduler:
    def __init__(self, eps_start, eps_end, eps_decay):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps = 0

    def value(self):
        eps = self.eps_end + (self.eps_start - self.eps_end) * \
              np.exp(-1.0 * self.steps / self.eps_decay)
        return eps

    def step(self):
        self.steps += 1
