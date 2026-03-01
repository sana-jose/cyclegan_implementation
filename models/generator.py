import torch.nn as nn
class Conv_block_k(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,norm=True,activation=True):
        super().__init__()
        layers=[]
        layers.append(nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding=kernel_size//2,padding_mode='reflect'))
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels,affine=True))
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.block=nn.Sequential(*layers)
    def forward(self,x):
        return self.block(x)
    
class Residual_block(nn.Module):
    def __init__(self,channels):
        super().__init__()
        layers=[]
        layers.append(Conv_block_k(channels,channels,3,1,norm=True,activation=True))
        layers.append(Conv_block_k(channels,channels,3,1,norm=True,activation=False))
        self.block=nn.Sequential(*layers)
    def forward(self,x):
        return x+self.block(x)
    
class Up_sample_block(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,norm=True,activation=True):
        super().__init__()
        layers=[]
        layers.append(nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding=kernel_size//2,output_padding=stride-1))
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels,affine=True))
        if activation:
            layers.append(nn.ReLU(inplace=True))
        self.block=nn.Sequential(*layers)
    def forward(self,x):
        return self.block(x)
        
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        layers=[]
        layers.append(Conv_block_k(3,64,7,1,norm=True,activation=True))
        layers.append(Conv_block_k(64,128,3,2,norm=True,activation=True))
        layers.append(Conv_block_k(128,256,3,2,norm=True,activation=True))
        for _ in range(9):
            layers.append(Residual_block(256))
        layers.append(Up_sample_block(256,128,3,2,norm=True,activation=True))
        layers.append(Up_sample_block(128,64,3,2,norm=True,activation=True))
        layers.append(Conv_block_k(64,3,7,1,norm=False,activation=False))
        layers.append(nn.Tanh())
        self.model=nn.Sequential(*layers)
    def forward(self,x):
        return self.model(x)
    

        

    
