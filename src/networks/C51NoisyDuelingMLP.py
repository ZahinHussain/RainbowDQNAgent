import torch.nn as nn
import torch.nn.functional as F
from networks.custom_layers.noisy_linear import NoisyLinear

class C51NoisyDuelingMLP(nn.Module):
    def __init__(self, obs_dim, num_actions, atom_size, support):
        super().__init__()

        self.num_actions = num_actions
        self.atom_size = atom_size
        self.support = support

        hidden = 128

        self.feature = nn.Sequential(
            NoisyLinear(obs_dim, hidden),
            nn.ReLU(),
            NoisyLinear(hidden, hidden),
            nn.ReLU(),
        )

        self.advantage = nn.Sequential(
            NoisyLinear(hidden, num_actions * atom_size),
        )

        self.value = nn.Sequential(
            NoisyLinear(hidden, atom_size),
        )

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x):
        batch_size = x.size(0)

        x = self.feature(x)

        adv = self.advantage(x).view(batch_size, self.num_actions, self.atom_size)
        val = self.value(x).view(batch_size, 1, self.atom_size)

        q_atoms = val + adv - adv.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=2)
        return dist.clamp(min=1e-8)