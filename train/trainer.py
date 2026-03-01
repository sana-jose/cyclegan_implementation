from loss.adverserial_loss import Adverserial_loss
from loss.cyclic_loss import Cyclic_loss
from loss.identity_loss import Identity_Loss
from train.train_step import train_step
import torch
import os
from tqdm import tqdm

class Trainer():
    def __init__(self,device,G, F, D_X, D_Y,lambda_cycle,lambda_identity):
        self.device=device
        self.G=G.to(device)
        self.F=F.to(device)
        self.D_X=D_X.to(device)
        self.D_Y=D_Y.to(device)
        if torch.cuda.device_count() > 1:
            self.G = torch.nn.DataParallel(self.G)
            self.F = torch.nn.DataParallel(self.F)
            self.D_X = torch.nn.DataParallel(self.D_X)
            self.D_Y = torch.nn.DataParallel(self.D_Y)
        self.adverserial_loss=Adverserial_loss().to(device)
        self.cyclic_loss=Cyclic_loss().to(device)
        self.identity_loss=Identity_Loss().to(device)
        self.optimizer_G=torch.optim.Adam(list(self.G.parameters())+list(self.F.parameters()),lr=0.0002,betas=(0.5,0.999))
        self.optimizer_D=torch.optim.Adam(list(self.D_X.parameters())+list(self.D_Y.parameters()),lr=0.0002,betas=(0.5,0.999))
        self.lambda_cycle=lambda_cycle
        self.lambda_identity=lambda_identity
        self.use_amp=(self.device.type=='cuda')
        self.scalar_G=torch.amp.GradScaler('cuda') if self.use_amp else None
        self.scalar_D = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.start_epoch=0

    def unwrap_model(self,model):
        if isinstance(model, torch.nn.DataParallel):
            return model.module
        return model
    
    def save_checkpoint(self,epoch,path):
        os.makedirs(path,exist_ok=True)
        torch.save({
            'G_state_dict': self.unwrap_model(self.G).state_dict(),
            'F_state_dict': self.unwrap_model(self.F).state_dict(),
            'D_X_state_dict': self.unwrap_model(self.D_X).state_dict(),
            'D_Y_state_dict': self.unwrap_model(self.D_Y).state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'epoch': epoch
        }, f"{path}/checkpoint_epoch_{epoch+1}.pth")

    def load_checkpoint(self,checkpoint_path):
        checkpoint=torch.load(checkpoint_path,map_location=self.device)
        self.unwrap_model(self.G).load_state_dict(checkpoint['G_state_dict'])
        self.unwrap_model(self.F).load_state_dict(checkpoint['F_state_dict'])
        self.unwrap_model(self.D_X).load_state_dict(checkpoint['D_X_state_dict'])
        self.unwrap_model(self.D_Y).load_state_dict(checkpoint['D_Y_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.start_epoch=checkpoint['epoch'] + 1
    
    def train(self,dataloader,epochs,save_checkpoint_path):
        start_epoch=self.start_epoch
        if start_epoch >= epochs:
            print("Training already completed.")
            return
        if start_epoch > 0:
            print(f"Resuming training from epoch {start_epoch}...")
        for epoch in range(start_epoch, epochs):
            self.unwrap_model(self.G).train()
            self.unwrap_model(self.F).train()
            self.unwrap_model(self.D_X).train()
            self.unwrap_model(self.D_Y).train()
            loop= tqdm(enumerate(dataloader),total=len(dataloader),desc=f"Epoch [{epoch+1}/{epochs}]")
            for i, data in loop:
                real_X = data['monet'].to(self.device)
                real_Y = data['original'].to(self.device)
                losses = train_step(self.adverserial_loss,self.cyclic_loss,self.identity_loss,self.G,self.F,self.D_X,self.D_Y,real_X,real_Y,self.optimizer_G,self.optimizer_D,self.use_amp,self.scalar_G,self.scalar_D,self.lambda_cycle,self.lambda_identity)
                loop.set_postfix(loss_G=losses['total_loss_G'], loss_F=losses['total_loss_F'], loss_D_X=losses['total_loss_D_X'], loss_D_Y=losses['total_loss_D_Y'])
            self.save_checkpoint(epoch, save_checkpoint_path)
    

    
    
