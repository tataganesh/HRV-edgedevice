import torch
from networks.regressor_circular import *
from networks.classifier_circular import *
from networks.net import *

# As shown https://discuss.pytorch.org/t/finding-model-size/130275



def find_model_details(model_path):
    model = torch.load(model_path)
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    model_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    size_all_mb = (param_size + buffer_size) / 1024
    # torch.save(model.state_dict(), 'ggg.json')
    print(f'model size: {size_all_mb:.5f}KB, total params: {model_total_params}')
    


find_model_details('/Users/ganesh/UofA/SNN/HRV-edgedevice/models/upsampler/2022-07-10-21:40:37/upsampler_2022-07-10-21:40:37.pt')





