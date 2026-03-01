from datasets import load_dataset
from torch.utils.data import Dataset
import io
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from data.transforms import get_train_transforms,get_test_transforms

class MonetDataset(Dataset):
    def __init__(self,split='train'):
        self.dataset=load_dataset("huggan/monet2photo", split=split)
        if split == "train":
            self.transform = get_train_transforms()
        else:
            self.transform = get_test_transforms()
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self,idx):
        item=self.dataset[idx]
        image_monet=Image.open(io.BytesIO(item['imageA']['bytes'])).convert('RGB')
        image_original=Image.open(io.BytesIO(item['imageB']['bytes'])).convert('RGB')
        image_monet=self.transform(image_monet)
        image_original=self.transform(image_original)
        return{
            'monet':image_monet,
            'original':image_original
        }



