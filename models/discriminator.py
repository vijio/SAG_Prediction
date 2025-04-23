class Discriminator(nn.Module):
    """Implements paper's discriminator architecture (Sec 3.3, 4)"""
    def __init__(self, in_dim, hidden_dim=1024):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # RUL prediction head with uncertainty (Eq.8-9)
        self.rul_head = BayesianLayer(hidden_dim, 2)
        
        # Adversarial discrimination head
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch=None)
        
        # Uncertainty-aware prediction
        rul_params = self.rul_head(x)
        mean, logvar = rul_params[:, 0], rul_params[:, 1]
        rul = mean + torch.exp(0.5*logvar) * torch.randn_like(mean)
        
        validity = self.adv_head(x)
        return rul, validity