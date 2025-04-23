def train_step(real_data):
    # Train Discriminator
    optimizer_D.zero_grad()
    
    # Generate fake data
    z = torch.randn(batch_size, config.latent_dim)
    fake_data = generator(z)
    
    # Discriminator forward
    real_pred, real_feat = discriminator(real_data)
    fake_pred, fake_feat = discriminator(fake_data.detach())
    
    # Calculate losses
    loss_real = -real_pred.mean()
    loss_fake = fake_pred.mean()
    gp = gradient_penalty(real_data, fake_data)
    loss_D = loss_real + loss_fake + config.loss.gradient_penalty_weight * gp
    
    # Update discriminator
    loss_D.backward()
    optimizer_D.step()
    
    # Train Generator every n_critic steps
    if global_step % config.n_critic == 0:
        optimizer_G.zero_grad()
        
        # Generator forward
        fake_pred, fake_feat = discriminator(fake_data)
        loss_G = -fake_pred.mean()
        
        # Feature matching
        if config.loss.fe_matching_weight > 0:
            fm_loss = 0
            for r_f, f_f in zip(real_feat, fake_feat):
                fm_loss += F.l1_loss(r_f.detach(), f_f)
            loss_G += config.loss.fe_matching_weight * fm_loss
            
        loss_G.backward()
        optimizer_G.step()