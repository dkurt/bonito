import torch
from bonito.util import load_model

model = load_model('dna_r9.4.1', 'cpu', weights='1', half=False)

# input = torch.randn(1, 1, 1000)
input = torch.randn(1, 1, 1, 1000)

# ~/.local/lib/python3.6/site-packages/torch/nn/modules/module.py
# if len(param.shape) == 0 and len(input_param.shape) == 1:
#     input_param = input_param[0]
#
##### NEW CODE BLOCK BEGIN #######################################################
# if len(param.shape) == 4:
#     param = param.reshape(param.shape[0], param.shape[1], param.shape[3])
##### NEW CODE BLOCK END #######################################################
#
# if input_param.shape != param.shape:
#     # local shape should match the one in checkpoint
#     error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
#
with torch.no_grad():
    torch.onnx.export(model, input, 'dna_r9.4.1_2d.onnx', input_names=['input'], output_names=['output'])
