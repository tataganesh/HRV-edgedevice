import torch
from torch.fft import fft,ifft
import torch.nn as nn
from torch.autograd import grad


# class CirConvNet(nn.Module):
#     def __init__(self,layer_sizes, dropout = 0):
#         super().__init__()
    
#         self.w1 = nn.Parameter(torch.randn(layer_sizes[0]))
#         self.w2 = nn.Parameter(torch.randn(layer_sizes[1]))
#         # self.w3 = nn.Parameter(torch.randn(layer_sizes[1]))
        
#         self.w3 = nn.Linear(layer_sizes[1], layer_sizes[2])
#         self.batchnorm1 = nn.BatchNorm1d(layer_sizes[2])
#         self.w4 = nn.Linear(layer_sizes[2], 1)
#         self.dropout_p = dropout
       
#         # self.w4= nn.Linear(layer_sizes[2], layer_sizes[3])
#         # self.w5= nn.Linear(layer_sizes[3], 1)
#     def forward(self, x):
#         # circular convolution followed by relu
#         x = torch.real(ifft(fft(x,dim=1,norm="ortho")*fft(self.w1,norm="ortho"),dim=1,norm="ortho"))
#         x = torch.sin(x)
#         # another circular convolution followed by relu
#         # x = torch.real(ifft(fft(x,dim=1,norm="ortho")*fft(self.w2,norm="ortho"),dim=1,norm="ortho"))
#         # x = torch.sin(x)
       
#         # Apply last linear layer
#         # x = self.w2(x)
#         # x = torch.sin(x)
#         # x = torch.dropout(x, 0.5, train=True)
        
#         x = self.w3(x)
#         x = torch.sin(x)
#         # x = self.batchnorm1(x)
#         x = torch.dropout(x, self.dropout_p, train=True)
#         y = self.w4(x)
#         # x = torch.sin(x)
#         # y = self.w4(x)
        
#         return y
    


# class CirConvNet(nn.Module):
#     def __init__(self,layer_sizes):
#         super().__init__()
#         self.layer1 = nn.Linear(layer_sizes[0], 53)
#         self.layer2 = nn.Linear(53, 25)
#         self.layer3 = nn.Linear(25, 1)
#     def forward(self, x):
#         x = torch.sin(self.layer1(x))
#         x = torch.dropout(x, 0.1, train=True)
#         x = torch.sin(self.layer2(x))
#         x = torch.dropout(x, 0.1, train=True)
#         y = self.layer3(x)
#         return y


# class CirConvNet(nn.Module):
#     def __init__(self,layer_sizes):
#         super().__init__()
#         self.layer1 = torch.nn.Conv1d(1, 10, 7)
#         self.layer2 = torch.nn.Conv1d(10, 10, 7)
#         self.layer3 = torch.nn.Linear(1010, 1)
#         # self.layer4 = torch.nn.Linear(10, 1)
#     def forward(self, x):
#         x = torch.unsqueeze(x, 1)
#         x = torch.sin(self.layer1(x))
#         # print(x.shape)
#         # exit()
#         # x = torch.dropout(x, 0.1, train=True)
#         x = torch.sin(self.layer2(x))
#         x = torch.flatten(x, 1)
#         y = self.layer3(x)
#         return y
    

# class CirConvNet(nn.Module):
#     def __init__(self,layer_sizes, dropout = 0):
#         super().__init__()
#         self.len = layer_sizes[0]
#         self.w1 = nn.Parameter(torch.randn(layer_sizes[1]))
#         self.w2 = nn.Parameter(torch.randn(layer_sizes[2]))
#         # print(layer_sizes)
#         # self.w3 = nn.Parameter(torch.randn(layer_sizes[1]))
        
#         self.w3 = nn.Linear(layer_sizes[0], layer_sizes[3])
#         # self.batchnorm1 = nn.BatchNorm1d(layer_sizes[2])
#         self.w4 = nn.Linear(layer_sizes[3], 1)
#         self.dropout_p = dropout
       
#         # self.w4= nn.Linear(layer_sizes[2], layer_sizes[3])
#         # self.w5= nn.Linear(layer_sizes[3], 1)
#     def forward(self, x):
#         # circular convolution followed by relu
#         x = torch.real(ifft(fft(x,dim=1,norm="ortho")*fft(self.w1,n=self.len,norm="ortho"),dim=1,norm="ortho"))
#         x = torch.relu(x)
#         # another circular convolution followed by relu
#         x = torch.real(ifft(fft(x,dim=1,norm="ortho")*fft(self.w2,n=self.len,norm="ortho"),dim=1,norm="ortho"))
#         x = torch.relu(x)
       
#         # Apply last linear layer
#         # x = self.w2(x)
#         # x = torch.sin(x)
#         # x = torch.dropout(x, 0.5, train=True)
        
#         x = self.w3(x)
#         x = torch.relu(x)
#         # x = self.batchnorm1(x)
#         x = torch.dropout(x, self.dropout_p, train=True)
#         y = self.w4(x)
#         # x = torch.sin(x)
#         # y = self.w4(x)
        
#         return y
    

# class CirConvNet(nn.Module):
#     def __init__(self,layer_sizes, dropout = 0):
#         super().__init__()
#         self.len = layer_sizes[0]
#         self.circ_conv_layers = list()
#         for layer_size in layer_sizes[1:]:
#             self.circ_conv_layers.append(nn.Parameter(torch.randn(layer_size)))
#         self.circ_conv_layers = nn.ParameterList(self.circ_conv_layers)
#         self.conv1 = nn.Conv2d(1, 1, 3)
#         self.w4 = nn.Linear(385, 1)
#         self.dropout_p = dropout
        
#     def forward(self, x):
#         # circular convolution followed by relu
#         circ_conv_ops = list()
#         for circ_conv_layer in self.circ_conv_layers:
#             op = torch.real(ifft(fft(x,dim=1,norm="ortho")*fft(circ_conv_layer,n=self.len,norm="ortho"),dim=1,norm="ortho"))
#             op = torch.relu(op)
#             circ_conv_ops.append(op)
#         stacked_layers = torch.stack(circ_conv_ops)
#         stacked_layers = torch.unsqueeze(stacked_layers, dim=0)
#         stacked_layers = torch.permute(stacked_layers, [2, 0, 3, 1])
#         op = self.conv1(stacked_layers)
#         op = torch.flatten(op, 1)
#         op = torch.relu(op)
#         y = self.w4(op)
#         return y

class CirConvNet(nn.Module):
    def __init__(self,layer_sizes, dropout = 0):
        super().__init__()
        self.len = layer_sizes[0]
        self.circ_conv_layers = list()
        for layer_size in layer_sizes[1:]:
            self.circ_conv_layers.append(nn.Parameter(torch.randn(layer_size)))
        self.circ_conv_layers = nn.ParameterList(self.circ_conv_layers)
        self.conv1 = nn.Conv1d(len(layer_sizes[1:]), 1, 3)
        self.w4 = nn.Linear(55, 1)
        
        self.dropout_p = dropout
        
    def forward(self, x):
        # circular convolution followed by relu
        circ_conv_ops = list()
        for circ_conv_layer in self.circ_conv_layers:
            op = torch.real(ifft(fft(x,dim=1,norm="ortho")*fft(circ_conv_layer,n=self.len,norm="ortho"),dim=1,norm="ortho"))
            op = torch.relu(op)
            circ_conv_ops.append(op)
        stacked_layers = torch.stack(circ_conv_ops)
        stacked_layers = torch.permute(stacked_layers, [1, 0, 2])
        op = self.conv1(stacked_layers)
        op = torch.flatten(op, 1)
        op = torch.relu(op)
        
        y = self.w4(op)
        # x = torch.sin(x)
        # y = self.w4(x)
        
        return y