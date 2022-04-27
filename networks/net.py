import torch
import torch.nn as nn
# NN Definition
class NeuralNetwork(torch.nn.Module):
    def __init__(self, n_inp_features, n_output_features, layer_sizes=[500, 700]):
        super(NeuralNetwork,self).__init__()
        self.layers = list()
        layer_sizes.insert(0, n_inp_features)
        # for i in range(1, len(layer_sizes)):
        #     self.layers.append(nn.Linear(layer_sizes[i-1], layer_sizes[i]))
        # self.layers = nn.ModuleList([l for l in self.layers])
        # self.dropout1 = nn.Dropout(p=0.25)
        # # self.layer2 = nn.Linear(60, 60)
        # # self.dropout2 = nn.Dropout(p=0.25)
        # self.layer3 = nn.Linear(40, n_output_features)
        self.hidden_layers = nn.ModuleList([nn.Linear(layer_sizes[i-1], layer_sizes[i])  for i in range(1, len(layer_sizes))])
        self.output_layer = nn.Linear(layer_sizes[-1], n_output_features)
    def forward(self,x):
        # x=self.layer1(x)
        # x = torch.relu(x)
        # # x = self.layer2(x)
        # # x = torch.relu(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = torch.relu(x)
            # x = torch.dropout(x, p=0.1, train=True)
        x = self.output_layer(x)
        return x

# class NeuralNetwork(torch.nn.Module):
#     def __init__(self, n_inp_features, n_output_features):
#         super(NeuralNetwork,self).__init__()

#         self.layer1 = nn.Linear(n_inp_features, 500)
#         self.dropout1 = nn.Dropout(p=0.25)
#         self.layer2 = nn.Linear(500, 700)
#         self.dropout2 = nn.Dropout(p=0.25)
#         self.layer3 = nn.Linear(700, n_output_features)
#         # self.layer3 = nn.Linear(num_input_features, 10)
#     def forward(self,x):
#         x=self.layer1(x)
#         # x = self.dropout1(x)
#         # x =(x)
#         x = torch.relu(x)
#         x = self.layer2(x)
#         # x = self.dropout2(x)
#         x = torch.relu(x)
#         x = self.layer3(x)
#         return x
