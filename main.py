from train.trainer import Trainer
from models.generator import Generator
from models.discriminator import Discriminator
from utils.init_weights import init_weights
from data.dataset import MonetDataset
from torch.utils.data import DataLoader
import sys
import torch
def main(save_checkpoint_path,load_checkpoint_path,num_workers):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G=Generator()
    F=Generator()
    D_x=Discriminator()
    D_y=Discriminator()
    lambda_cycle=10.0
    lambda_identity=5.0
    init_weights(G)
    init_weights(F)
    init_weights(D_x)
    init_weights(D_y)
    trainer=Trainer(device,G,F,D_x,D_y,lambda_cycle,lambda_identity)
    print("Models initialized and Trainer created on device:", device)
    if load_checkpoint_path:
        trainer.load_checkpoint(load_checkpoint_path)
        print(f"Checkpoint loaded from {load_checkpoint_path}")
    dataset=MonetDataset()
    dataloader=DataLoader(dataset,batch_size=1,shuffle=True,num_workers=4)
    trainer.train(dataloader=dataloader,epochs=200,save_checkpoint_path=save_checkpoint_path)
    

if __name__ == "__main__":
    load_checkpoint_path=sys.argv[1] if sys.argv[1] != "None" else None
    save_checkpoint_path=sys.argv[2]
    num_workers=int(sys.argv[3]) if len(sys.argv) > 3 else 1
    main(save_checkpoint_path,load_checkpoint_path,num_workers)
    
