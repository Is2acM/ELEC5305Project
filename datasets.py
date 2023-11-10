'''
Script for dataset loading and transforming.

'''

import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# Specify dataset directory
rootdir = './Mel'
traindir = os.path.join(rootdir, 'train')
valdir = os.path.join(rootdir, 'val')

# batch size
Batch_size = 32
# Data transforms, can be used for data augmentation
Data_transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = ImageFolder(root=traindir, transform=Data_transform)
valid_dataset = ImageFolder(root=valdir, transform=Data_transform)

train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=Batch_size, shuffle=False)

classes = os.listdir(traindir)