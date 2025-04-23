import torch
import torch.nn as nn
import torch.nn.functional as F

class TimeDistributed(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        
    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        
        batch_size, timesteps = x.size()[:2]
        reshaped = x.view(-1, *x.size()[2:])
        output = self.module(reshaped)
        return output.view(batch_size, timesteps, *output.size()[1:])

class WaveGANDiscriminator(nn.Module):
    def __init__(self, in_channels, hidden_dims=[64, 128, 256, 512]):
        super().__init__()
        layers = []
        current_dim = in_channels
        for dim in hidden_dims:
            layers += [
                nn.Conv1d(current_dim, dim, 25, stride=4, padding=11),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ]
            current_dim = dim
            
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(current_dim, 1)
        
    def forward(self, x):
        features = []
        for layer in self.net:
            x = layer(x)
            features.append(x)
        x = x.mean(dim=-1)
        return self.fc(x), features