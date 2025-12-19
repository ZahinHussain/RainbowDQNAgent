import torch.nn as nn

class DuelingMLP(nn.Module):
    def __init__(self, obs_dim, num_actions):
        super().__init__()

        hidden = 128

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        self.advantage = nn.Linear(hidden, num_actions)
        self.value = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.feature(x)

        adv = self.advantage(x)          # [B, A]
        val = self.value(x)              # [B, 1]

        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q