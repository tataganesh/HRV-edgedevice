import torch
import torch.nn as nn
# NN Definition
class NeuralNetwork(torch.nn.Module):
    def __init__(self, n_inp_features, n_output_features):
        super(NeuralNetwork,self).__init__()
        self.layer1 = nn.Linear(n_inp_features, 500)
        self.dropout1 = nn.Dropout(p=0.25)
        self.layer2 = nn.Linear(500, 700)
        self.dropout2 = nn.Dropout(p=0.25)
        self.layer3 = nn.Linear(700, n_output_features)
    def forward(self,x):
        x=self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        return x
