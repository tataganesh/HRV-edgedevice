import torch
from torch.fft import fft,ifft
import torch.nn as nn
from torch.autograd import grad


class CirConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        num_channels = 5
        kernel_size = 5
        self.layer1 = torch.nn.Conv1d(1, num_channels, kernel_size)
        self.layer2 = torch.nn.Conv1d(num_channels, num_channels, kernel_size)
        self.layer3 = torch.nn.Linear(305, 1)
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = torch.sin(self.layer1(x))
        x = torch.sin(self.layer2(x))
        x = torch.flatten(x, 1)
        y = self.layer3(x)
        return y


class ConvNet2D(nn.Module):
    def __init__(self):
        super().__init__()
        num_channels = 5
        kernel_size = 5
        self.layer1 = torch.nn.Conv2d(1, num_channels, (5,1))
        self.layer2 = torch.nn.Conv2d(num_channels, num_channels, (5,1))
        self.layer3 = torch.nn.Linear(305, 1)
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = torch.unsqueeze(x, -1)
        x = torch.sin(self.layer1(x))
        x = torch.sin(self.layer2(x))
        x = torch.flatten(x, 1)
        y = self.layer3(x)
        return y


class HRNet(nn.Module):
    def __init__(self,nLayers,hiddenN,nIn,nOut):
        super(HRNet, self).__init__()
        layers = []
        layers.append(torch.nn.Linear(nIn, hiddenN))
        layers.append(torch.nn.ReLU())
        for i in range(nLayers):
            layers.append(torch.nn.Linear(hiddenN, hiddenN))
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Linear(hiddenN, nOut))
        
        self.net = nn.Sequential(*layers)
                    
    def forward(self,x):
        return self.net(x)