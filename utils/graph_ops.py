import torch
from torch import nn

def spectral_norm(module):
    return nn.utils.spectral_norm(module) if module.training else module

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=nn.LeakyReLU(0.2)):
        super().__init__()
        self.residual = nn.Sequential(
            spectral_norm(nn.Conv1d(in_channels, out_channels, 3, padding=1)),
            activation,
            spectral_norm(nn.Conv1d(out_channels, out_channels, 3, padding=1))
        )
        self.shortcut = spectral_norm(nn.Conv1d(in_channels, out_channels, 1)) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

class AttentionLayer(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel_attn = nn.Sequential(
            spectral_norm(nn.Conv1d(channel, channel//8, 1)),
            nn.ReLU(),
            spectral_norm(nn.Conv1d(channel//8, channel, 1)),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        attn = self.channel_attn(x.mean(dim=-1, keepdim=True))
        return x * attn
