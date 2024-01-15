import os
#import gc
#import glob
#import json
#from collections import defaultdict
#from collections import OrderedDict
#import multiprocessing as mp
from pathlib import Path
#from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple
#import warnings

from tqdm import tqdm
import matplotlib.pyplot as plt
#import matplotlib.patches as patches
import numpy as np
#import pandas as pd
import PIL.Image as Image
#from sklearn.metrics import fbeta_score
#from sklearn.exceptions import UndefinedMetricWarning
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as thd
from torch.nn.utils import clip_grad_norm_
from torch.optim import lr_scheduler
from torchvision import transforms
from torchvision import disable_beta_transforms_warning
from torchvision.transforms import v2 
#import torchvision.transforms.v2 as v2
import lightning as light
from torchmetrics.classification import BinaryFBetaScore, Dice
#import torch.nn.functional as F

#from lightning.pytorch import  LightningModule, Trainer #LightningDataModule,
#from lightning.pytorch.cli import LightningCLI
#import pytorch_lightning as pl


from model import Model12
#from util import Dataset2

Image.MAX_IMAGE_PIXELS = None


             
fabric = light.Fabric(accelerator="auto", devices="auto")
fabric.launch()


class SubvolumeDataset(thd.Dataset):
    def __init__(
        self,
        fragments: List[Path],
        voxel_shape: Tuple[int, int, int],
        load_inklabels: bool = True,
        filter_edge_pixels: bool = False,

    ):
        self.fragments = sorted(map(lambda path: path.resolve(), fragments))
        self.image_stack = []
        self.labels = []
        self.fragmentsWidths = []
        self.fragmentsHeights = []
        self.nmbrOfTiles = []
        self.tile_height = 16
        self.tile_width = 16
        self.tile_channels = 15 #16


        for fragment_id, fragment_path in enumerate(self.fragments):
            fragment_path = fragment_path.resolve()  # absolute path
            mask = np.array(Image.open(str(fragment_path / "mask.png")).convert("1"))
            

            surface_volume_paths = sorted(
                (fragment_path / "surface_volume").rglob("*.tif")
            )
  
            
            # 3D inputs: no convert to torch yet, since it doesn't support uint16
            images = []
            if fragment_path.name == "2":
                for fn in surface_volume_paths[16:46]:
                    fullImage = np.array(Image.open(fn))
                    if filter_edge_pixels:
                        fullImage = fullImage * mask
                    slicedImage = fullImage[:, :].copy()#---------[0:7000, 0:5000]
                    images.append(slicedImage)
            elif fragment_path.name == "5":
                for fn in surface_volume_paths[16:46]:
                    fullImage = np.array(Image.open(fn))
                    if filter_edge_pixels:
                        fullImage = fullImage * mask
                    slicedImage = fullImage[286:5480, 1120:4394].copy()#---------
                    images.append(slicedImage)
            elif fragment_path.name == "6":
                for fn in surface_volume_paths[16:]:
                    fullImage = np.array(Image.open(fn))
                    if filter_edge_pixels:
                        fullImage = fullImage * mask
                    #slicedImage = fullImage[437:5020, 670:4341].copy()#---------
                    
                    slicedImage1 = fullImage[4541:5030, 837:1560].copy() #s
                    slicedImage2 = fullImage[697:1233, 3755:4478].copy() #a
                    slicedImage3 = np.concatenate((slicedImage1, slicedImage2), axis=0)
                    
                    slicedImage1 = fullImage[1700:2725, 412:1533].copy() #pi
                    slicedImage2 = np.concatenate((slicedImage1, slicedImage3), axis=1)
                    
                    slicedImage1 = fullImage[400:2673, 1713:3557].copy()
                    slicedImage = np.concatenate((slicedImage1, slicedImage2), axis=0)
                    images.append(slicedImage)
            elif fragment_path.name == "7":
                for fn in surface_volume_paths[1:]:
                    fullImage = np.array(Image.open(fn))
                    if filter_edge_pixels:
                        fullImage = fullImage * mask
                    #slicedImage = fullImage[400:2900, 1180:11050].copy()#---------
                    
                    slicedImage1 = fullImage[980:1771, 9317:9960].copy() #p
                    slicedImage2 = fullImage[1145:1936, 10206:11153].copy() #pi
                    slicedImage3 = fullImage[2103:2894, 10409:10788].copy() #i
                    slicedImage4 = fullImage[1674:2465, 8395:9003].copy() #g
                    slicedImage5 = fullImage[380:1171, 5181:6640].copy() #2a
                    slicedImage6 = fullImage[835:1626, 1152:1740].copy() #n
                    slicedImage = np.concatenate((slicedImage6, slicedImage5, slicedImage1, slicedImage2, slicedImage4, slicedImage3), axis=1)
                    images.append(slicedImage)
            elif fragment_path.name == "8":
                for fn in surface_volume_paths[1:]:
                    fullImage = np.array(Image.open(fn))
                    if filter_edge_pixels:
                        fullImage = fullImage * mask
                    #slicedImage = fullImage[1900:9500, 1250:8210].copy()#---------
                    slicedImage1 = fullImage[8600:9723, 5599:8427].copy()
                    slicedImage2 = fullImage[1620:2743, 5723:7194].copy()
                    slicedImage3 = fullImage[7300:8423, 1047:2155].copy()
                    slicedImage = np.concatenate((slicedImage1, slicedImage2, slicedImage3), axis=1)
                    images.append(slicedImage)
            elif fragment_path.name == "9":
                for fn in surface_volume_paths[1:]:
                    fullImage = np.array(Image.open(fn))
                    if filter_edge_pixels:
                        fullImage = fullImage * mask
                    #slicedImage = fullImage[1177:9313, 752:3722].copy()#---------
                    
                    slicedImage1 = fullImage[1137:1693, 1385:2075].copy() #pi                   
                    slicedImage2 = fullImage[1724:3112, 745:1435].copy() #pii
                    slicedImage3 = fullImage[2128:2324, 1613:2303].copy() #i
                    slicedImage4 = fullImage[3215:3745, 933:1623].copy() #u
                    slicedImage5 = fullImage[3746:4534, 1293:1983].copy() #s
                    slicedImage6 = fullImage[7557:8141, 2033:2723].copy() #pi
                    slicedImage7 = fullImage[8494:9378, 2205:2895].copy() #o
                    slicedImage8 = fullImage[8338:8898, 3133:3823].copy() #pi
                    
                    slicedImage = np.concatenate((slicedImage1, slicedImage3, slicedImage2, slicedImage4, slicedImage5, slicedImage6, slicedImage7, slicedImage8), axis=0)
                    images.append(slicedImage)
            elif fragment_path.name == "a1":
                for fn in surface_volume_paths[1:]:
                    fullImage = np.array(Image.open(fn))
                    if filter_edge_pixels:
                        fullImage = fullImage * mask
                    #slicedImage = fullImage[837:11030, 370:3570].copy()#---------

                    slicedImage1 = fullImage[760:5817, 369:2705].copy()  
                    
                    slicedImage2 = fullImage[7633:9058, 2550:3653].copy() #pi
                    slicedImage3 = fullImage[9745:11170, 2637:3870].copy() #ga
                    slicedImage4 = np.concatenate((slicedImage2, slicedImage3), axis=1)
                            
                    slicedImage = np.concatenate((slicedImage1, slicedImage4), axis=0)
                    images.append(slicedImage)
            elif fragment_path.name == "a2":
                for fn in surface_volume_paths[1:]:
                    fullImage = np.array(Image.open(fn))
                    if filter_edge_pixels:
                        fullImage = fullImage * mask
                    slicedImage = fullImage[325:6350, 1100:].copy()#---------
                    images.append(slicedImage)
            elif fragment_path.name == "a3":
                for fn in surface_volume_paths[1:]:
                    fullImage = np.array(Image.open(fn))
                    if filter_edge_pixels:
                        fullImage = fullImage * mask
                    slicedImage = fullImage[2418:26235, 204:5152].copy()#---------
                    images.append(slicedImage)
            elif fragment_path.name == "a4":
                for fn in surface_volume_paths[1:]:
                    fullImage = np.array(Image.open(fn))
                    if filter_edge_pixels:
                        fullImage = fullImage * mask
                    slicedImage = fullImage[:2835, 3509:9415].copy()#---------
                    images.append(slicedImage)
            elif fragment_path.name == "a5":
                for fn in surface_volume_paths[1:]:
                    fullImage = np.array(Image.open(fn))
                    if filter_edge_pixels:
                        fullImage = fullImage * mask

                    slicedImage1 = fullImage[62:1128, 9757:11117].copy()  #pie
                    slicedImage2 = fullImage[53:1119, 11506:12556].copy() #v
                    slicedImage3 = fullImage[239:1305, 13989:15395].copy() #pi
                    slicedImage4 = fullImage[803:1869, 4720:5601].copy() #a
                    slicedImage5 = fullImage[1941:3007, 10776:11658].copy() #c
                    slicedImage6 = fullImage[2437:3503, 13427:14290].copy() #y
                    slicedImage7 = np.concatenate((slicedImage1, slicedImage2, slicedImage3, slicedImage4, slicedImage5, slicedImage6), axis=1)

                    images.append(slicedImage)
            elif fragment_path.name == "b1":
                for fn in surface_volume_paths[1:]:
                    fullImage = np.array(Image.open(fn))
                    if filter_edge_pixels:
                        fullImage = fullImage * mask
                    images.append(fullImage)
            elif fragment_path.name == "b2":
                for fn in surface_volume_paths[1:]:
                    fullImage = np.array(Image.open(fn))
                    if filter_edge_pixels:
                        fullImage = fullImage * mask
                    images.append(fullImage)
            else:
                for fn in surface_volume_paths[16:46]:
                    fullImage = np.array(Image.open(fn))
                    if filter_edge_pixels:
                        fullImage = fullImage * mask
                    slicedImage = fullImage[: , :].copy()#---------
                    images.append(slicedImage)

            images = np.stack(images, axis=0)  
            
           

            channels, img_height, img_width = images.shape
            tile_height = self.tile_height 
            tile_width = self.tile_width
            tile_channels = self.tile_channels 
            img_divisibleHeight = img_height + (16-img_height%tile_height)
            img_divisibleWidth = img_width + (16-img_width%tile_width)

            images = np.pad(images, ((0, 0), (0, img_divisibleHeight-img_height), (0, img_divisibleWidth-img_width)), 'constant', constant_values=((0,0),(0,0),(0,0)))

            self.fragmentsWidths.append(img_divisibleWidth)
            self.fragmentsHeights.append(img_divisibleHeight)
            self.nmbrOfTiles.append((img_divisibleWidth*img_divisibleHeight)//(self.tile_height*self.tile_width))

            #images = images.reshape(channels, 
            #                        img_divisibleHeight // tile_height, tile_height,
            #                        img_divisibleWidth // tile_width, tile_width, 
            #                        )
            #
            #images = images.transpose(1,3,0,2,4)
            #images = images.reshape(-1, tile_channels, tile_height, tile_width)
            self.image_stack.append(images)
     
                    
            # labels: binary
            inklabels = (
                np.array(Image.open(str(fragment_path / "inklabels.png")).convert("1")) > 0
            )
            if fragment_path.name == "2":
                inklabels = inklabels[:, :]#---------[0:7000, 0:5000]
            elif fragment_path.name == "5":
                inklabels = inklabels[286:5480, 1120:4394]#---------
            elif fragment_path.name == "6":
                #inklabels = inklabels[437:5020, 670:4341]#---------

                inklabels1 = inklabels[4541:5030, 837:1560].copy() #s
                inklabels2 = inklabels[697:1233, 3755:4478].copy() #a
                inklabels3 = np.concatenate((inklabels1, inklabels2), axis=0)
                    
                inklabels1 = inklabels[1700:2725, 412:1533].copy() #pi
                inklabels2 = np.concatenate((inklabels1, inklabels3), axis=1)
                    
                inklabels1 = inklabels[400:2673, 1713:3557].copy()
                inklabels = np.concatenate((inklabels1, inklabels2), axis=0)         
            elif fragment_path.name == "7":
                #inklabels = inklabels[400:2900, 1180:11050]#---------
                inklabels1 = inklabels[980:1771, 9317:9960].copy() #p
                inklabels2 = inklabels[1145:1936, 10206:11153].copy() #pi
                inklabels3 = inklabels[2103:2894, 10409:10788].copy() #i
                inklabels4 = inklabels[1674:2465, 8395:9003].copy() #g
                inklabels5 = inklabels[380:1171, 5181:6640].copy() #2a
                inklabels6 = inklabels[835:1626, 1152:1740].copy() #n
                inklabels = np.concatenate((inklabels6, inklabels5, inklabels1, inklabels2, inklabels4, inklabels3), axis=1)
            elif fragment_path.name == "8":
                #inklabels = inklabels[1900:9500, 1250:8210]#---------
                inklabels1 = inklabels[8600:9723, 5599:8427].copy()
                inklabels2 = inklabels[1620:2743, 5723:7194].copy()
                inklabels3 = inklabels[7300:8423, 1047:2155].copy()
                inklabels = np.concatenate((inklabels1, inklabels2, inklabels3), axis=1)
            elif fragment_path.name == "9":
                #inklabels = inklabels[1177:9313, 752:3722]#---------

                inklabels1 = inklabels[1137:1693, 1385:2075].copy() #pi                   
                inklabels2 = inklabels[1724:3112, 745:1435].copy() #pii
                inklabels3 = inklabels[2128:2324, 1613:2303].copy() #i
                inklabels4 = inklabels[3215:3745, 933:1623].copy() #u
                inklabels5 = inklabels[3746:4534, 1293:1983].copy() #s
                inklabels6 = inklabels[7557:8141, 2033:2723].copy() #pi
                inklabels7 = inklabels[8494:9378, 2205:2895].copy() #o
                inklabels8 = inklabels[8338:8898, 3133:3823].copy() #pi                  
                inklabels = np.concatenate((inklabels1, inklabels3, inklabels2, inklabels4, inklabels5, inklabels6, inklabels7, inklabels8), axis=0)        
            elif fragment_path.name == "a1":
                #inklabels = inklabels[837:11030, 370:3570]#---------
                inklabels1 = inklabels[760:5817, 369:2705].copy()  
                    
                inklabels2 = inklabels[7633:9058, 2550:3653].copy() #pi
                inklabels3 = inklabels[9745:11170, 2637:3870].copy() #ga
                inklabels4 = np.concatenate((inklabels2, inklabels3), axis=1)
                            
                inklabels = np.concatenate((inklabels1, inklabels4), axis=0)   
            elif fragment_path.name == "a2":
                inklabels = inklabels[325:6350, 1100:]#---------
            elif fragment_path.name == "a3":
                inklabels = inklabels[2418:26235, 204:5152]#---------
                plt.imshow(inklabels)
                plt.show()
            elif fragment_path.name == "a4":
                inklabels = inklabels[:2835, 3509:9415]#---------
                plt.imshow(inklabels)
                plt.show()
            elif fragment_path.name == "a5":
                inklabels1 = inklabels[62:1128, 9757:11117].copy()  #pie
                inklabels2 = inklabels[53:1119, 11506:12556].copy() #v
                inklabels3 = inklabels[239:1305, 13989:15395].copy() #pi
                inklabels4 = inklabels[803:1869, 4720:5601].copy() #a
                inklabels5 = inklabels[1941:3007, 10776:11658].copy() #c
                inklabels6 = inklabels[2437:3503, 13427:14290].copy() #y
                inklabels = np.concatenate((inklabels1, inklabels2, inklabels3, inklabels4, inklabels5, inklabels6), axis=1)
            else:
                inklabels = inklabels[:,:]#---------

           
        

            # labels: as 16*16 tiles:
            inklabels = np.pad(inklabels, ((0, img_divisibleHeight-img_height), (0, img_divisibleWidth-img_width)), 'constant', constant_values=((0,0),(0,0)))
            
            inklabels = inklabels.reshape( 
                                    img_divisibleHeight // tile_height, tile_height,
                                    img_divisibleWidth // tile_width, tile_width, 
                                    )
            inklabels = inklabels.transpose(0,2,1,3)
            inklabels = inklabels.reshape(-1, tile_height, tile_width)
            self.labels.append(inklabels)

            print(f"Loaded fragment {fragment_path} on {os.getpid()}")
            #gc.collect()
            

        
        self.labels = np.concatenate(self.labels)             
        self.nmbrLabels = self.labels.shape[0]
        
      
    def __len__(self):
        return self.nmbrLabels

    def __getitem__(self, index):
        labelIndex = index

        # coordinate transformations (from N(z)*16*16 labels coordinate system):
        z = 0
        while self.nmbrOfTiles[z] <= index:
            index = index-self.nmbrOfTiles[z]
            z = z + 1
            

        b = (self.fragmentsWidths[z]//16)
        a = index // b
        y = a * 16
        x = (index - b*a) * 16
        

        # extra pad at borders (for d*224*224 tensor input tiles):
        ymin = y - 112
        ymax = y + 112
        xmin = x - 112
        xmax = x + 112

        add_ymin = 0
        add_ymax = 0
        add_xmin = 0
        add_xmax = 0
        
      
        if ymin < 0: 
            add_ymin = ymin * -1
            
        if ymax > self.fragmentsHeights[z]: 
            add_ymax = ymax - self.fragmentsHeights[z] 
            
        if xmin < 0: 
            add_xmin = xmin * -1
            
        if xmax > self.fragmentsWidths[z]: 
            add_xmax = xmax - self.fragmentsWidths[z]
            

        # inputs: as d*224*224 tiles:
        inputs = self.image_stack[z][:, ymin+add_ymin:ymax-add_ymax, add_xmin+xmin:xmax-add_xmax].astype(np.float32)   
        inputs = np.pad(inputs, ((0, 0), (add_ymin, add_ymax), (add_xmin, add_xmax)), 'constant', constant_values=((0,0),(0,0),(0,0)))
        
            
        return inputs / 65535, self.labels[labelIndex] # converted to tensors by pytorch

    


base_path = Path("path") 
train_path = base_path / "train"
all_fragments = sorted([f.name for f in train_path.iterdir()])
train_fragments = [train_path / fragment_name for fragment_name in all_fragments]
print("All fragments:", train_fragments)



torch.backends.cudnn.benchmark = True

train_dset = SubvolumeDataset(fragments=train_fragments, voxel_shape=(30, 224, 224), filter_edge_pixels=True)
print("Num items (pixels)", len(train_dset))

BATCH_SIZE = 32
train_loader = thd.DataLoader(train_dset, batch_size=BATCH_SIZE, drop_last=True, shuffle=True, pin_memory=True)
print("Num batches:", len(train_loader))




model = Model12()
checkpoint = fabric.load("kaggle/working/model.pt")

model.load_state_dict(checkpoint['model'])
model.train()
model.modelNv.train()


# pytorch augmentation
train_transform = v2.RandomApply([
v2.RandomVerticalFlip(p=0.5),
v2.RandomHorizontalFlip(p=0.5),
v2.RandomApply([v2.RandomChoice([
    v2.RandomRotation(degrees=(90, 90)),
    v2.RandomRotation(degrees=(180, 180)),
    v2.RandomRotation(degrees=(-90, -90))
    ])],0.75),
v2.RandomSolarize(threshold=0.4, p=0.2),
v2.RandomInvert(p=0.3),    
v2.RandomApply([v2.RandomOrder([
    v2.RandomPosterize(bits=3, p=0.2),
    v2.RandomPosterize(bits=4, p=0.2),
    v2.RandomEqualize(p=0.3),
    v2.RandomAutocontrast(p=0.3),
    v2.RandomAdjustSharpness(sharpness_factor=0, p=0.2),
    v2.RandomAdjustSharpness(sharpness_factor=0.5, p=0.2),
    v2.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
    v2.RandomApply([v2.RandAugment()],0.2),
    v2.RandomApply([v2.AugMix()],0.2),
    v2.RandomApply([v2.AutoAugment(v2.AutoAugmentPolicy.IMAGENET)],0.2),
    v2.RandomApply([v2.AutoAugment(v2.AutoAugmentPolicy.CIFAR10)],0.2),
    v2.RandomApply([v2.AutoAugment(v2.AutoAugmentPolicy.SVHN)],0.2),
    v2.RandomApply([v2.TrivialAugmentWide()],0.2),
    v2.RandomPhotometricDistort(p=0.2),
    v2.RandomApply([v2.RandomAffine(degrees=(0, 0), scale=(0.1, 4.0), shear=(-70,70))],0.15),
    v2.RandomApply([v2.RandomAffine(degrees=(0, 0), scale=(0.1, 4.0))],0.15),
    v2.RandomPerspective(p=0.2),
    v2.RandomApply([v2.ColorJitter(brightness=(0.2, 0.9), hue=(-0.4,0.4), contrast=(0.2, 0.9), saturation=(0.2, 0.9))],0.2)
    ])
                          ],0.8),
v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 4.0))], 0.4),
v2.RandomApply([v2.ElasticTransform()], 0.4),
v2.RandomPhotometricDistort(p=0.2),
v2.RandomApply([v2.ColorJitter(brightness=(0.1, 1.0), hue=(-0.5,0.5), contrast=(0.1, 1.0), saturation=(0.1, 1.0))],0.4),
v2.RandomApply([v2.RandomChoice([transforms.Compose([
    v2.RandomRotation(degrees=(-180, 180), expand=True),
    v2.Resize((224,224), antialias=False) 
        ]),  
    v2.RandomRotation(degrees=(-180, 180), expand=False)
                           ])
    ],0.6)           
],0.8)


def applyTransform(tensor): 
    for j in range(tensor.shape[0]): 
        tensor[j] = train_transform(tensor[j])
    




learning_rate = 1e-5

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01, amsgrad=True)
scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9) 
metric = fabric.to_device(BinaryFBetaScore(beta=2.0, threshold = 0.5))
criterion = nn.BCELoss(reduction='mean')

model, optimizer = fabric.setup(model, optimizer)
train_loader = fabric.setup_dataloaders(train_loader) 


#accumSteps = 32 / BATCH_SIZE   
running_loss = 0
running_accuracy = 0.0
running_fbeta = 0.0
denom = 0

for i, (subvolumes, inklabels) in tqdm(enumerate(train_loader), total=len(train_loader)):
    
    # apply augmentation (on GPU)
    labels_transformed = v2.Pad((104, 104, 104, 104), padding_mode = "constant")(inklabels)       
    subvolumes_labels = torch.cat((subvolumes, labels_transformed.unsqueeze(1)), 1)      
    applyTransform(subvolumes_labels.unsqueeze(2))
    
    
    inklabels = subvolumes_labels[:,-1:,104:120, 104:120].squeeze(1).flatten(1).gt(0).float()  
    outputs = model(subvolumes_labels[:,:-1]).sigmoid()
    bInklabels = inklabels.bool()
    bOutputs = outputs.bool()
        

    # label smoothing
    outRound = outputs.round()
    outputs = outputs-(outputs - outRound)*((outRound == bInklabels) & (outputs.gt(0.9) | outputs.lt(0.2))) 
        

    loss = criterion(outputs,  inklabels) + (1-metric(outputs, bInklabels))
       
    fabric.backward(loss)

    #clip_grad_norm_(model.parameters(), 0.5) #T.nn.utils.clip_grad_value_(net.parameters, clip_value=1.5)

    #if (i + 1) % accumSteps == 0:
    optimizer.step()
            
    optimizer.zero_grad()
          
        
           

    accuracy = (bOutputs == bInklabels) 
    running_accuracy += accuracy
    running_loss += loss
    denom += 1
        
    
    if (i+1) % 1000 == 0:     
        if scheduler._last_lr[0] < 7e-6:
            optimizer.param_groups[0]["lr"] = learning_rate
                

        print("Nr.", i+1)
            
        print(outputs)
        print(inklabels)
            

        print("nmbrOfOnes:", inklabels.bool().sum().item())
        print("accuracy:",(outputs.gt(0.5).int() == inklabels.int()).float().mean())      
        print("loss:", loss.item())
        print("AccuracyM:", running_accuracy.float().mean().item() / denom)  
        print("lossM:", running_loss.item() / denom) 
        print("metric:", metric(outputs, inklabels.bool()))
        print("Nr.", i+1)
        print(optimizer.param_groups[0]["lr"])
     
        running_loss = 0
        running_accuracy = 0.
        running_fbeta = 0.
        denom = 0


    if (i+1) % 1000 == 0:                
        torch.save({'model': model.state_dict()},'kaggle/working/model.pt')
        print("************************save*************************************")
        print("************************save************************************") 
        
        scheduler.step()
        
            
