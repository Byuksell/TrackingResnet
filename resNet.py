# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 20:54:11 2020

@author: Buket
"""

from __future__ import print_function
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet152, resnet18
import torchvision
import pandas as pd
import numpy as np
from img_to_vec import getVectorDataloader
import torchvision.models as models
from torch.utils.data import DataLoader
from os import listdir
from PIL import Image
import math
import scipy as sp
from collections import namedtuple
from enum import IntEnum
from enum import Enum
import matplotlib.pyplot as plt
from datetime import datetime 

resnet = models.resnet152(pretrained=True)
cuda = True if torch.cuda.is_available() else False
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
ByteTensor=torch.cuda.ByteTensor if cuda else torch.ByteTensor
device=("cuda" if torch.cuda.is_available() else "cpu") 
batch=1
EPOCHS=100
learning_rate = 1e-3
resLoss=open("resLoss.txt","w+",1)


def load_image(infilename):
   
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="float32")
        
    return data


class loadTrainData(torch.utils.data.Dataset):
       
    def __init__(self):
        
        # LOAD FRAMES
        data_transform = transforms.Compose([transforms.ToTensor()])
    
        self.images = torch.empty(0,3,480,640)
        self.img_vecs = torch.empty(0,2048)
        self.prev_images = torch.empty(0,3,480,640)        
        self.img_prev_vec=torch.empty(0,2048)
        
        self.gt= torch.empty(0,4)
        self.gt_prev= torch.empty(0,4)
   
        #data_path_list = ["/scratch/users/byuksel13/Tracking_batch/OTB/Box_new/Box/img2"]   
        #pathGT_list = ["/scratch/users/byuksel13/Tracking_batch/OTB/Box/Box/Box-center.csv"]     
        
        data_path_list = ["/scratch/users/byuksel13/Tracking_ResNet/OTB/Box_deneme/Box/img2"]   
        pathGT_list = ["/scratch/users/byuksel13/Tracking_ResNet/OTB/Box_deneme/Box/box_norm.csv"] 
        #print("img loading")
        
        for f in range(0,len(data_path_list )):        
            data_path = data_path_list[f]
            image_folder = torchvision.datasets.ImageFolder(root=data_path, transform=data_transform)        
            images = torch.empty(len(image_folder),3,480,640)
            
            for i in range(0,len(image_folder)):
                temp = image_folder[i][0].unsqueeze(0)     # images
                images[i] = temp
                
            prev_images = torch.empty(len(image_folder),3,480,640)
            
            for m in range(0,len(image_folder)):
                temp2 = image_folder[m][0].unsqueeze(0)    #prev_images
                prev_images[m] = temp2
            
                                

            img_vectors = torch.empty(len(images),2048)
            for j in range(0,len(images)):             
                temp = getVectorDataloader(images[j].unsqueeze(0))      # img_vecs
                img_vectors[j] = temp
                
            
            prev_img_vectors=torch.empty(len(images),2048)            
            for k in range(0,len(images)):                    
                temp = getVectorDataloader(prev_images[k].unsqueeze(0))     #prev_img_vecs
                prev_img_vectors[k] = temp
                
            pathGT = pathGT_list[f]                
            gt_prev=pd.read_csv(pathGT, dtype=np.float32, header=None)        #gt_prev
            gt_prev= gt_prev.values
            gt_prev=torch.from_numpy(gt_prev)
            
            gt=pd.read_csv(pathGT, dtype=np.float32, header=None)             #gt
            gt= gt.values
            gt=torch.from_numpy(gt)
            
           
           

            self.images = torch.cat((self.images,images[1:]),0)
            self.img_vecs = torch.cat((self.img_vecs,img_vectors[1:]),0)
            self.img_prev_vec=torch.cat((self.img_prev_vec,prev_img_vectors[0:]),0)
            self.gt = torch.cat((self.gt,gt[1:]),0)
            self.gt_prev = torch.cat((self.gt_prev,gt_prev[0:-1]),0)
   
        #print("img loaded")
       
        self.len = len(self.images) 
       
        
    def __getitem__(self,index):
            
        return self.images[index],self.img_vecs[index],self.img_prev_vec[index],self.gt[index],self.gt_prev[index]
        
    def __len__(self):
            
        return self.len
    
     
       
    
    
def train_layers(loss_fn,optimizer,fc,img_prev_vecF,img_vecF,gt_prev):
    optimizer.zero_grad()
    gt_prev.view(-1,1)
    fc_in= torch.cat((img_prev_vecF, img_vecF, gt_prev), 1)  
    prediction=fc(fc_in) 
    loss=loss_fn(prediction,gt)     
    print(prediction,"",gt)
    loss.backward()
    optimizer.step()
    return loss



dataset_trn = loadTrainData()
dataloader_trn = torch.utils.data.DataLoader(dataset=dataset_trn,batch_size=batch,shuffle=False)
#print(dataloader_trn.dataset)
fc = nn.Sequential(nn.Linear(4100, 300), nn.ReLU(), nn.Linear(300, 4))
fc=fc.to(device)
optimizer = torch.optim.Adam(fc.parameters(),lr=learning_rate)
loss_fn = torch.nn.MSELoss(reduction='mean').to(device)
for epoch in range(EPOCHS):
    
    totalLoss=0       

    resLoss.write("Epoch:%d " %epoch)
             
#-------------------------------TRAIN MAIN LOOP------------------------  
    for i,(img,img_vec,img_prev_vec,gt,gt_prev) in enumerate (dataloader_trn): 
        img_vec = img_vec.type(FloatTensor)
        img_prev_vec=img_prev_vec.type(FloatTensor)
        gt = gt.type(FloatTensor)
        gt_prev = gt_prev.type(FloatTensor)

        loss=train_layers(loss_fn,optimizer,fc,img_prev_vec, img_vec, gt_prev)  
        totalLoss+=loss    
    totalLoss=totalLoss/dataloader_trn.dataset.len
    resLoss.write("Loss: %f\n" %totalLoss)            


  
  