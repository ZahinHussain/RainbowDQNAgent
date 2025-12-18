from numpy import size
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Noise buffers (not learnable)
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        limit = 1 / math.sqrt(self.in_features)

        # Initialize μ (mean)
        self.weight_mu.data.uniform_(-limit, limit)
        self.bias_mu.data.uniform_(-limit, limit)

        # Initialize σ (std)
        sigma_weight = self.std_init / math.sqrt(self.in_features)
        sigma_bias = self.std_init / math.sqrt(self.out_features)

        self.weight_sigma.data.fill_(sigma_weight)
        self.bias_sigma.data.fill_(sigma_bias)

    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)

        # Outer product for weight noise
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        w = self.weight_mu + self.weight_sigma * self.weight_epsilon
        b = self.bias_mu + self.bias_sigma * self.bias_epsilon
        return F.linear(x, w, b)

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


