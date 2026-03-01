import torch.nn as nn
import torch
class Adverserial_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss=nn.MSELoss()

    def forward(self,predicted,real_or_fake):
        if real_or_fake:
            target=torch.ones_like(predicted)
        else:
            target=torch.zeros_like(predicted)
        return self.loss(predicted,target)