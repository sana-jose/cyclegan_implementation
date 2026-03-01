import torch
def train_step(adverserial_loss,cyclic_loss,identity_loss,generator_G,generator_F,discriminator_X,discriminator_Y,real_X,real_Y,optimizer_G,optimizer_D,use_amp,scalar_G,scalar_D,lambda_cycle,lambda_identity):
    optimizer_G.zero_grad()
    with torch.amp.autocast(device_type='cuda', enabled=use_amp):
        fake_y = generator_G(real_X)
        cycled_x = generator_F(fake_y)
        fake_x = generator_F(real_Y)
        cycled_y = generator_G(fake_x)
        same_x = generator_F(real_X)
        same_y = generator_G(real_Y)
        loss_adverserial_G = adverserial_loss(discriminator_Y(fake_y), True)
        loss_cycle_G= cyclic_loss(real_X, cycled_x) * lambda_cycle
        loss_identity_G = identity_loss(real_Y, same_y) * lambda_identity
        total_loss_G = loss_adverserial_G + loss_cycle_G + loss_identity_G
        loss_adverserial_F = adverserial_loss(discriminator_X(fake_x), True)
        loss_cycle_F = cyclic_loss(real_Y, cycled_y) * lambda_cycle
        loss_identity_F = identity_loss(real_X, same_x) * lambda_identity
        total_loss_F = loss_adverserial_F + loss_cycle_F + loss_identity_F
        total_loss = total_loss_G + total_loss_F
    if use_amp:
        scalar_G.scale(total_loss).backward()
        scalar_G.step(optimizer_G)
        scalar_G.update()
    else:
        total_loss.backward()
        optimizer_G.step()
    optimizer_D.zero_grad()
    with torch.cuda.amp.autocast():
        loss_discriminator_Y_real = adverserial_loss(discriminator_Y(real_Y), True)
        loss_discriminator_Y_fake = adverserial_loss(discriminator_Y(fake_y.detach()), False)
        total_loss_D_Y = (loss_discriminator_Y_real + loss_discriminator_Y_fake)
        loss_discriminator_X_real = adverserial_loss(discriminator_X(real_X), True)
        loss_discriminator_X_fake = adverserial_loss(discriminator_X(fake_x.detach()), False)
        total_loss_D_X = (loss_discriminator_X_real + loss_discriminator_X_fake)
        total_loss_D = 0.5*(total_loss_D_Y + total_loss_D_X)
    if use_amp:
        scalar_D.scale(total_loss_D).backward()
        scalar_D.step(optimizer_D)
        scalar_D.update()
    else:
        total_loss_D.backward()
        optimizer_D.step()
    return {
        "total_loss_G": total_loss_G.item(),
        "total_loss_F": total_loss_F.item(),
        "total_loss_D_Y": total_loss_D_Y.item(),
        "total_loss_D_X": total_loss_D_X.item()
    }
