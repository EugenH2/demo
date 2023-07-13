import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import glob



class DatasetAll(torch.utils.data.Dataset):   
    #Characterizes a dataset for PyTorch
    def __init__(self, path, labels, transform=None):
        self.labels = labels
        #Load all images
        self.images = torch.stack(
            [transform(Image.open(i)) for i in glob.glob(path + "*.jpg")]
            ) 
        #.convert('L')


    def __len__(self):
        #Number of samples
        return len(self.labels)

    def __getitem__(self, index):
        #Generates one sample of data

        y = F.pad(torch.tensor(self.labels[index]), pad=(0,32-len(self.labels[index])), mode='constant', value=0).long()
            
        return self.images[index], y



class Dataset(torch.utils.data.Dataset):   
    #Characterizes a dataset for PyTorch
    def __init__(self, path, labels, transform=None):
        self.labels = labels
        self.path = path
        self.transform = transform

    def __len__(self):
        #Number of samples
        return len(self.labels)

    def __getitem__(self, index):
        #Generates one sample of data

        # Load data and get label
        X = Image.open(self.path + str(index) + ".jpg")#.convert('L')   
        X = self.transform(X)
  
        y = F.pad(torch.tensor(self.labels[index]), pad=(0,32-len(self.labels[index])), mode='constant', value=0).long()
        
        return X, y

