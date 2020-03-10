import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

device=("cuda" if torch.cuda.is_available() else "cpu") 
   
resnet = models.resnet152(pretrained=True)
    
for param in resnet.parameters():
  param.requires_grad = False

# Remove last layers of resnet
resnet.fc = nn.Sequential()

def getVectorDataloader(x):
    with torch.no_grad():
        return resnet(x)
    