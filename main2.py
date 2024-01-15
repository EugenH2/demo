#import torchvision
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch import nn
#import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn.utils import clip_grad_norm
import torchvision.transforms.v2 as transformsv2

from util import Dataset, DatasetAll
from model import VisionTransformer, LSTMModel
import numpy as np
import matplotlib.pyplot as plt




use_cuda = torch.cuda.is_available()
print(use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")
#torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True  #at::globalContext().setBenchmarkCuDNN(true);

# Parameters
train_transform = transforms.Compose([
transforms.Resize((32,256)), #(32,384)
transforms.ToTensor(),
#transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#transforms.Normalize((0.5, ), (0.5, ))
#transformsv2.RandAugment()
transformsv2.RandomApply([
transformsv2.RandomVerticalFlip(p=0.5),
transformsv2.RandomHorizontalFlip(p=0.5),
transformsv2.ColorJitter(brightness=(0.1, 1.0), hue=(-0.5,0.5), contrast=(0.1, 1.0), saturation=(0.1, 1.0)),
transformsv2.TrivialAugmentWide(),
transformsv2.AutoAugment(transformsv2.AutoAugmentPolicy.CIFAR10),
transformsv2.RandomPhotometricDistort(),
transformsv2.RandomChoice([transforms.Compose([
    transformsv2.RandomRotation(degrees=(-30, 30), expand=True),
    transforms.Resize((32,256), antialias=False) ]), 
    transformsv2.RandomRotation(degrees=(-30, 30), expand=False)
                           ])
    ],0.8)
])


max_epochs = 10000
learningRate = 1e-4



inputsPath  = "../out/trdg -c 100000 -w 1 -na 2/" #"trdg -c 100000 -w 1 -na 2, ten"
labelsPath  = "../out/trdg -c 100000 -w 1 -na 2/labels.txt"


# Data Generators
labels = []
with open(labelsPath, "r") as file:
    for line in file:
        labels.append([ord(i)*1 for i in line.strip()[line.find(" ")+1:]])


train_set = Dataset(inputsPath, labels, train_transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, pin_memory=True)


# Choose a model:
model = VisionTransformer(
        embed_dim = 256,
        hidden_dim = 1024,
        num_heads = 8,
        num_layers = 6,
        patch_size = 4,
        num_channels = 3,
        num_classes = 4096,
        dropout = 0.3)

model = LSTMModel(num_classes = 4096)


model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b0', pretrained=True)
model.classifier[3] = nn.Sequential(
    nn.Linear(1280, 2560),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(2560, 4096))



model.to(device)
model.train()


#Loading weights
#checkpoint = torch.load('save/model.pt', map_location=device)
#model.load_state_dict(checkpoint['model'])
#optimizer.load_state_dict(checkpoint['optimizer'])


criterion = nn.CrossEntropyLoss(reduction='sum')
optimizer = optim.AdamW(model.parameters(), lr=learningRate,  eps=1e-08, weight_decay=0.01, amsgrad=True) #betas=(0.5, 0.999)
#optimizer = optim.SGD(model.parameters(), lr=learningRate, momentum = 0.9, nesterov = True)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)



# Loop over epochs

for epoch in range(max_epochs):
    # Training
    running_loss = 0.0
    running_accuracy = 0.0
    #running_fbeta = 0.0
    denom = 0

    
    for i, (inputs,labels) in enumerate(train_loader):
  
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        outputs = outputs.view(outputs.shape[0],32,128)
        
        loss = criterion(outputs.view(-1,outputs.size(-1)), labels.view(-1))
    
        loss.backward()

        #clip_grad_norm_(model.parameters(), 0.4)

        
        optimizer.step()
        optimizer.zero_grad()    
        
        
        #running_fbeta += metric(outputs, labels)
        running_accuracy += (outputs.argmax(dim=-1) == labels).float().mean()
        running_loss += loss
        denom += 1
        print(loss)
        
        if (i+1) % 1000 == 0:
            print("Nr.", i+1)           
            print(outputs.argmax(dim=-1))
            print(labels)
            
            
            print("accuracy:",(outputs.argmax(dim=-1) == labels).float().mean())    
            print("loss:", loss)
            print("running_accuracy:", running_accuracy / denom)
            print("running_loss:", running_loss.item() / denom)
            #print("running_fbeta:", running_fbeta / denom)      
            
            print("Nr.", i+1)
            print("lr:", optimizer.param_groups[0]["lr"])
     
            running_loss = 0.
            running_accuracy = 0.
            #running_fbeta = 0.
            denom = 0


        if (i+1) % 1000 == 0:
            torch.save({'model': model.state_dict(),'optimizer': optimizer.state_dict(),},'model.pt')
            print("****************************save************************************")
            print("****************************save************************************") 
            
            scheduler.step()
        
        

            
