import torch
import torch.nn as nn
# NN Definition
class NeuralNetwork(torch.nn.Module):
    def __init__(self, n_inp_features, n_output_features, layer_sizes=[500, 700]):
        super(NeuralNetwork,self).__init__()
        self.layers = list()
        layer_sizes.insert(0, n_inp_features)
        self.hidden_layers = nn.ModuleList([nn.Linear(layer_sizes[i-1], layer_sizes[i])  for i in range(1, len(layer_sizes))])
        self.output_layer = nn.Linear(layer_sizes[-1], n_output_features)
    def forward(self,x):
        for layer in self.hidden_layers:
            x = layer(x)
            x = torch.relu(x)
        x = self.output_layer(x)
        return x
