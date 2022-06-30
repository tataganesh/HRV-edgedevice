from inspect import stack
import torch
from torch.fft import fft,ifft
import torch.nn as nn
from torch.autograd import grad
import matplotlib.pyplot as plt

class CirConvNet(nn.Module):
    def __init__(self,layer_sizes, dropout = 0):
        super().__init__()
        self.len = layer_sizes[0]
        self.circ_conv_layers = list()
        for layer_size in layer_sizes[1:]:
            self.circ_conv_layers.append(nn.Parameter(torch.randn(layer_size)))
        self.circ_conv_layers = nn.ParameterList(self.circ_conv_layers)
        self.w4 = nn.Linear(layer_sizes[0], 1)
        self.dropout_p = dropout
        
    def forward(self, x):
        # circular convolution followed by relu
        for circ_conv_layer in self.circ_conv_layers:
            x = torch.real(ifft(fft(x,dim=1,norm="ortho")*fft(circ_conv_layer,n=self.len,norm="ortho"),dim=1,norm="ortho"))
            x = torch.relu(x)
        
        y = self.w4(x)
        # x = torch.sin(x)
        # y = self.w4(x)
        
        return y
    

class CirConvNetStacked2d(nn.Module):
    def __init__(self,layer_sizes, dropout = 0):
        super().__init__()
        self.len = layer_sizes[0]
        self.circ_conv_layers = list()
        for layer_size in layer_sizes[1:]:
            self.circ_conv_layers.append(nn.Parameter(torch.randn(layer_size)))
        self.circ_conv_layers = nn.ParameterList(self.circ_conv_layers)
        self.conv1 = nn.Conv2d(1, 1, 3)
        self.w4 = nn.Linear(378, 1)
        self.dropout_p = dropout
        
    def forward(self, x):
        # circular convolution followed by relu
        circ_conv_ops = list()
        for circ_conv_layer in self.circ_conv_layers:
            op = torch.real(ifft(fft(x,dim=1,norm="ortho")*fft(circ_conv_layer,n=self.len,norm="ortho"),dim=1,norm="ortho"))
            op = torch.relu(op)
            circ_conv_ops.append(op)
        stacked_layers = torch.stack(circ_conv_ops)
        stacked_layers = torch.unsqueeze(stacked_layers, dim=0)
        stacked_layers = torch.permute(stacked_layers, [2, 0, 3, 1])
        op = self.conv1(stacked_layers)
        op = torch.flatten(op, 1)
        op = torch.relu(op)
        y = self.w4(op)
        return y
    
class CirConvNetStacked1d(nn.Module):
    def __init__(self,layer_sizes, dropout = 0):
        super().__init__()
        self.len = layer_sizes[0]
        self.circ_conv_layers = list()
        for layer_size in layer_sizes[1:]:
            self.circ_conv_layers.append(nn.Parameter(torch.randn(layer_size)))
        self.circ_conv_layers = nn.ParameterList(self.circ_conv_layers)
        self.conv1 = nn.Conv1d(len(layer_sizes[1:]), 1, 3)
        self.w4 = nn.Linear(67, 1)
        
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
    
    
if __name__ == "__main__":
    activation = dict()
    def print_shape(self, inp, outp):
        print('Inside ' + self.__class__.__name__ + ' forward')
        print('')
        print('input: ', type(inp))
        print('input[0]: ', type(inp[0]))
        print('output: ', type(outp))
        print('')
        print('input size:', inp[0].size())
        print('output size:', outp.data.size())
        activation["conv2"] = outp.detach()
    net = CirConvNetStacked2d([56, 56, 56, 56, 56])
    net.conv1.register_forward_hook(print_shape)
    x = torch.randn((1, 56))
    net(x)
    print(activation['conv2'].numpy())
    plt.imshow(activation['conv2'].numpy().reshape(54, 2))
    plt.show()
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(pytorch_total_params)


