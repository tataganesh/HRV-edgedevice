from inspect import stack
import torch
from torch.fft import fft,ifft
import torch.nn as nn
from torch.autograd import grad
import matplotlib.pyplot as plt

    
class CirConvNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Conv1d(1, 5, 5)
        self.layer2 = torch.nn.Conv1d(5, 5, 5)
        # self.layer3 = torch.nn.Conv1d(5, 3, 3)
        self.layer4 = torch.nn.Linear(305, 1)
    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        # x = torch.relu(self.layer3(x))
        x = torch.flatten(x, 1)
        y = self.layer4(x)
        y = torch.sigmoid(y)
        return y

class FCNClassifier(torch.nn.Module):
    def __init__(self, n_inp_features, n_output_features, layer_sizes=[500, 700]):
        super(FCNClassifier,self).__init__()
        self.layers = list()
        layer_sizes.insert(0, n_inp_features)
        self.hidden_layers = nn.ModuleList([nn.Linear(layer_sizes[i-1], layer_sizes[i])  for i in range(1, len(layer_sizes))])
        self.output_layer = nn.Linear(layer_sizes[-1], n_output_features)
    def forward(self,x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = torch.sin(x)
        y = self.output_layer(x)
        y = torch.sigmoid(y)
        return y


