import torch.nn.functional as F
import torch
import torch.nn as nn

class scatEncoder(nn.Module):
    def __init__(self):
        super(scatEncoder,self).__init__()

        self.mp = nn.MaxPool2d(2)
        self.drop = nn.Dropout(0.4)

        self.conv1_1 = nn.Conv2d(417,256,3,padding=1)
        self.conv1_2 = nn.Conv2d(256,256,3,padding=1)

        self.conv2_1 = nn.Conv2d(256,512,3,padding=0)
        self.conv2_2 = nn.Conv2d(512,768,3,padding=0)

    def forward(self,x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.mp(x)

        x = F.relu(self.conv2_2(F.relu(self.conv2_1(x))))
        x = self.drop(x)
        x = self.mp(x)
        return x
