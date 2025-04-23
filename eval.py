import torch
from utils.metrics import calculate_fid, mmd_rbf

def evaluate_model(generator, test_loader, device):
    real_features = []
    fake_features = []
    
    with torch.no_grad():
        for batch in test_loader:
            real_data = batch.to(device)
            z = torch.randn(real_data.size(0), config.latent_dim).to(device)
            fake_data = generator(z)
            
            real_features.append(feature_extractor(real_data))
            fake_features.append(feature_extractor(fake_data))
            
    real_features = torch.cat(real_features).cpu().numpy()
    fake_features = torch.cat(fake_features).cpu().numpy()
    
    return {
        'fid': calculate_fid(real_features, fake_features),
        'mmd': mmd_rbf(real_features, fake_features)
    }