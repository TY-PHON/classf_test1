import torch
import os
import torch.nn as nn
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
import torchvision.transforms as transforms


#data augmentations 
#test aug
test_tfm = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])
#train aug
train_tfm = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
])

class FoodDataset(Dataset):

    def __init__(self,path,tfm=test_tfm,files = None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        #test load image
        print("one {path} sample",self.files[0])
        self.transform = tfm
    
    #what's the meaning?
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)

        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1
        return im,label
    
    

        
