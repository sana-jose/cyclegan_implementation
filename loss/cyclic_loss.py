import torch.nn as nn
import torch
class Cyclic_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss=nn.L1Loss()
    
    def forward(self,original,reconstructed):
        return self.loss(original,reconstructed)