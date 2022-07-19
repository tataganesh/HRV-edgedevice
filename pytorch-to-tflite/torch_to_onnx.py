import torch
import sys
sys.path.append('../')
from networks.net import NeuralNetwork
from networks.regressor_circular import CirConvNet, HRNet, ConvNet2D
from networks.classifier_circular import CirConvNetClassifier, fcn_classifier
import argparse

activations = dict()
def conv_inp_op(self, inp, outp):
    activations["input"] = inp[0].detach()
    activations["output"] = outp.detach()

parser = argparse.ArgumentParser(description='ANN Upsampler Training')
parser.add_argument('--pytorch_model_path', help="Path to pytorch model", required=True)
parser.add_argument('--batch_size', help="Set Batch Size", default=1, type=int)
parser.add_argument('--input_size', help="Input Size", default=35, type=int)


args = parser.parse_args()

onnx_model_path = 'model.onnx'

model = torch.load(args.pytorch_model_path)
model.eval()

sample_input = torch.rand((args.batch_size, args.input_size))

y = model(sample_input)

torch.onnx.export(
    model,
    sample_input, 
    onnx_model_path,
    verbose=False,
    input_names=['input'],
    output_names=['output'],
    opset_version=12
)