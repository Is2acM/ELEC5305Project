'''

-------------------------------------------
'''
import torch
import torch.nn as nn
from datasets import classes

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64*54*54, 128)
        self.dropout2 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, len(classes))

        
    def forward(self, x):
        x = self.dropout1(self.pool(self.leaky_relu(self.conv1(x))))
        x = self.pool(self.leaky_relu(self.conv2(x)))
        # print(x.size())
        x = x.view(-1, 64*54*54) # Flatten Layer
        x = self.dropout2(self.fc1(x))
        x = self.fc2(x)
        return x

# class CNNModel(nn.Module): # 60 # Modified
#     def __init__(self):
#         super(CNNModel, self).__init__()
#         # define the network layers
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.relu1 = nn.ReLU()
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.relu2 = nn.ReLU()
#         self.conv3 = nn.Conv2d(16, 32, 3)
#         self.relu3 = nn.ReLU()
#         self.fc1 = nn.Linear(32 * 5 * 5, 120)
#         self.relu4 = nn.ReLU()
#         self.fc2 = nn.Linear(120, 84)
#         self.relu5 = nn.ReLU()
#         self.fc3 = nn.Linear(84, len(classes))
        
#     def forward(self, x):
#         x = self.pool(self.relu1(self.conv1(x)))
#         x = self.pool(self.relu2(self.conv2(x)))
#         x = self.pool(self.relu3(self.conv3(x)))
#         # print(x.size())
#         x = x.view(-1, 32 * 5 * 5)
#         x = self.relu4(self.fc1(x))
#         x = self.relu5(self.fc2(x))
#         x = self.fc3(x)
#         return x


def Network(train = False):
    if train:
        model = CNNModel()
    else:
        model.load_state_dict(torch.load("./outputs/model.pth", map_location='cuda'))
    return model

