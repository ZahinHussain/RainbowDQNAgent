import torch.nn as nn
import torch.nn.functional as F
from networks.custom_layers.noisy_linear import NoisyLinear

class C51NoisyDuelingCNN(nn.Module):
    def __init__(self, num_actions, atom_size, support):
        super().__init__()

        self.num_actions = num_actions
        self.atom_size = atom_size
        self.support = support

        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = 64 * 7 * 7

        # Advantage head
        self.advantage = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, num_actions * atom_size),
        )

        # Value head
        self.value = nn.Sequential(
            NoisyLinear(conv_out_size, 512),
            nn.ReLU(),
            NoisyLinear(512, atom_size),
        )

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x):

        batch_size = x.size(0)

        x = self.conv(x).view(batch_size, -1)

        adv = self.advantage(x).view(batch_size, self.num_actions, self.atom_size)

        val = self.value(x).view(batch_size, 1, self.atom_size)

        # Combine streams; subtract mean advantage for stability.
        q_atoms = val + adv - adv.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=2)
        return dist.clamp(min=1e-8)