import torch.nn as nn
import torch
class Identity_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss=nn.L1Loss()
    def forward(self,original,identity):
        return self.loss(original,identity)
