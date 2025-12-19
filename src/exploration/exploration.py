import numpy as np

class LinearEpsilonScheduler:
    def __init__(self, eps_start: float, eps_end: float, decay_steps: int):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.decay_steps = decay_steps

        self.epsilon = eps_start
        self.decay = (eps_start - eps_end) / decay_steps

    def value(self) -> float:
        return self.epsilon

    def step(self):
        self.epsilon = max(
            self.eps_end,
            self.epsilon - self.decay
        )

