# This script provides a method which builds OpenVINO network in runtime
import numpy as np
from openvino.inference_engine import IECore, IENetwork

import ngraph.opset4 as ng
from ngraph.impl.op import Parameter
from ngraph.impl import Function, Shape, Type

import torch
from torch.autograd import Variable


nodes = {}
out = None

def forward_hook(self, inputs, output):
    global out
    layer_type = self.__class__.__name__

    params = [value.numpy() for value in self.state_dict().values()]

    inp = nodes[inputs[0].data_ptr()]
    if layer_type == 'Conv1d':
        weights = np.expand_dims(params[0], axis=2)
        if self.groups == 1:
            out = ng.convolution(inp, weights,
                                 [1, self.stride[0]],
                                 [0, self.padding[0]],
                                 [0, self.padding[0]],
                                 [1, self.dilation[0]])

        else:
            weights = weights.reshape(self.groups, weights.shape[0] // self.groups, weights.shape[1], weights.shape[2], weights.shape[3])
            out = ng.group_convolution(inp, weights,
                                       [1, self.stride[0]],
                                       [0, self.padding[0]],
                                       [0, self.padding[0]],
                                       [1, self.dilation[0]])
        if len(params) > 1:
            assert(len(params) == 2)
            bias = params[1].reshape(1, params[1].shape[0], 1, 1)
            out = ng.add(out, bias)

    elif layer_type == 'BatchNorm1d':
        out = ng.batch_norm_inference(inp, params[0], params[1], params[2], params[3], self.eps)
    elif layer_type == 'Swish':
        out = ng.swish(inp)
    elif layer_type == 'Add':
        out = ng.add(inp, nodes[inputs[1].data_ptr()])
    elif layer_type == 'Dropout':
        return
    elif layer_type == 'Permute':
        order = []
        # 1D to 2D: i.e. (2, 0, 1) -> (2, 3, 0, 1)
        for d in self.dims:
            assert(d <= 2)
            order.append(d)
            if d == 2:
                order.append(3)
        out = ng.transpose(inp, order)
    else:
        raise Exception('Unknown layer type: ' + layer_type)

    nodes[output.data_ptr()] = out


def sanity_check(net, inp, ref):
    ie = IECore()
    exec_net = ie.load_network(net, 'CPU')
    ie_out = exec_net.infer({'input': inp.numpy()})
    ie_out = next(iter(ie_out.values()))

    ref = ref.numpy().reshape(ie_out.shape)
    diff = np.max(np.abs(ie_out - ref))
    print('PyTorch / OpenVINO diff:', diff)
    print('Reference values range: [{}, {}]'.format(np.min(ref), np.max(ref)))
    if diff > 1.1e-4:
        raise Exception('Sanity check failed with diff', diff)


def torch2openvino(model):
    with torch.no_grad():
        model.eval()
        hooks = []
        for module in model.modules():
            if len([m for m in module.modules()]) != 1:
                continue
            hooks.append(module.register_forward_hook(forward_hook))

        # Just a dummy input to make forward pass
        inp = Variable(torch.randn([1, 1, 1000]))

        param = Parameter(Type.f32, Shape([1, 1, 1, 1000]))
        nodes[inp.data_ptr()] = param
        ref = model(inp)

        for hook in hooks:
            hook.remove()

    out_node = ng.log(ng.softmax(out, axis=3))

    param.set_friendly_name('input')
    out_node.set_friendly_name('output')
    func = Function([out_node], [param], '')

    caps = Function.to_capsule(func)
    net = IENetwork(caps)

    # Uncomment to perform conversion check
    # sanity_check(net, inp, ref)

    return net
