from PIL import Image
import random
from torchvision import transforms
def random_jitter(image,resize_size=285,crop_size=256):
    image=image.resize((resize_size,resize_size))
    x=random.randint(0,resize_size-crop_size)
    y=random.randint(0,resize_size-crop_size)
    image=image.crop((x,y,x+crop_size,y+crop_size))
    if random.random()>0.5:
        image=image.transpose(Image.FLIP_LEFT_RIGHT)
    return image

def get_train_transforms():
    return transforms.Compose([random_jitter,transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

def get_test_transforms():
    return transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])