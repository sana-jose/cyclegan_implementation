import torch.nn as nn
class Conv_block_k_leaky(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,norm=True,activation=True):
        super().__init__()
        layers=[]
        layers.append(nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding=kernel_size//2,padding_mode='reflect'))
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels,affine=True))
        if activation:
            layers.append(nn.LeakyReLU(0.2,inplace=True))
        self.block=nn.Sequential(*layers)
    def forward(self,x):
        return self.block(x)
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        layers.append(Conv_block_k_leaky(3,64,4,2,norm=False,activation=True))
        layers.append(Conv_block_k_leaky(64,128,4,2,norm=True,activation=True))
        layers.append(Conv_block_k_leaky(128,256,4,2,norm=True,activation=True))
        layers.append(Conv_block_k_leaky(256,512,4,1,norm=True,activation=True))
        layers.append(nn.Conv2d(512,1,4,1,padding=4//2,padding_mode='reflect'))
        self.model=nn.Sequential(*layers)
    def forward(self,x):
        return self.model(x)
        