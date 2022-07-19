import torch
from torch.fft import fft,ifft
import torch.nn as nn
from torch.autograd import grad

class CirConvHRNet(nn.Module):
    def __init__(self,m,n):
        super().__init__()
        self.len = n
        self.w1 = nn.Parameter(torch.randn(m))
        self.w2 = nn.Parameter(torch.randn(m))
        # self.w3 = nn.Parameter(torch.randn(m))

        self.w3 = nn.Linear(n,1)
    def forward(self, x):
        # circular convolution followed by relu
        x = torch.real(ifft(fft(x,dim=1,norm="ortho")*fft(self.w1,n=self.len,norm="ortho"),dim=1,norm="ortho"))
        x = torch.relu(x)
        # another circular convolution followed by relu
        x = torch.real(ifft(fft(x,dim=1,norm="ortho")*fft(self.w2,n=self.len,norm="ortho"),dim=1,norm="ortho"))
        x = torch.relu(x)
        # Apply last linear layer
        
        # x = torch.real(ifft(fft(x,dim=1,norm="ortho")*fft(self.w3,n=self.len,norm="ortho"),dim=1,norm="ortho"))
        # x = torch.relu(x)
        y = self.w3(x)
        return y

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