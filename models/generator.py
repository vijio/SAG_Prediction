class Generator(nn.Module):
    """Implements paper's generator architecture (Sec 3.3)"""
    def __init__(self, noise_dim=200, hidden_dim=1024, out_dim=14, noise_std=0.1):
        super().__init__()
        self.noise_std = noise_std
        
        self.conv1 = GCNConv(noise_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            BayesianLayer(hidden_dim, out_dim)  # Paper's uncertainty quantification
        )

    def forward(self, z, edge_index):
        # Node-level noise injection (Sec 3.3)
        if self.training and self.noise_std > 0:
            z = z + torch.randn_like(z) * self.noise_std
        
        x = F.relu(self.conv1(z, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch=None)
        return self.fc(x)