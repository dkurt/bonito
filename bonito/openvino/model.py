import os
import io
import numpy as np
import torch

from bonito.nn import Swish

try:
    from openvino.inference_engine import IECore, StatusCode
    from .loader import convert_to_2d
except ImportError:
    pass

class OpenVINOModel:

    def __init__(self, model, half, dirname):
        self.model = model
        self.alphabet = model.alphabet
        self.parameters = model.parameters
        self.stride = model.stride

        package = model.config['model']['package']
        if not package in ['bonito.ctc', 'bonito.crf']:
            raise Exception('Unknown model configuration: ' + package)
        self.is_ctc = package == 'bonito.ctc'
        self.is_crf = package == 'bonito.crf'

        if self.is_crf:
            self.seqdist = model.seqdist
            self.encoder = lambda data : self(data, encoder=True)

        model_name = 'model' + ('_fp16' if half else '')
        xml_path, bin_path = [os.path.join(dirname, model_name) + ext for ext in ['.xml', '.bin']]
        self.ie = IECore()
        if os.path.exists(xml_path) and os.path.exists(bin_path):
            self.net = self.ie.read_network(xml_path, bin_path)
        else:
            # There is an issue with Swish at export step so we temporarly use default implementation
            origin_swish_forward = Swish.forward
            def swish_fake_forward(self, x):
                return x * torch.sigmoid(x)
            Swish.forward = swish_fake_forward

            if self.is_crf:
                inp = torch.randn([1, 1, 1000])
                torch.onnx.export(model.encoder, inp, os.path.join(dirname, model_name) + '.onnx',
                                  input_names=['input'], output_names=['output'],
                                  opset_version=11)
                raise Exception('OpenVINO 2021.2 is required to build CRF model in runtime. Use Model Optimizer instead.')

            # Just a dummy input for export
            inp = torch.randn([1, 1, 1, 1000])
            buf = io.BytesIO()

            # 1. Replace 1D layers to their 2D alternatives to improve efficiency
            convert_to_2d(model)

            # 2. Convert model to ONNX buffer
            torch.onnx.export(model, inp, buf, input_names=['input'], output_names=['output'],
                              opset_version=11)
            Swish.forward = origin_swish_forward

            # 3. Import network from memory buffer
            self.net = self.ie.read_network(buf.getvalue(), b'', init_from_buffer=True)

        self.exec_net = None


    def eval(self):
        pass


    def half(self):
        return self


    def to(self, device):
        self.device = str(device).upper()


    def __call__(self, data, encoder=False):
        data = data.float()
        if self.is_ctc:
            data = np.expand_dims(data, axis=2)  # 1D->2D
        batch_size = data.shape[0]
        inp_shape = list(data.shape)
        inp_shape[0] = 1  # We will run the batch asynchronously
        if self.net.input_info['input'].tensor_desc.dims != inp_shape:
            self.net.reshape({'input': inp_shape})
            self.exec_net = None
        if not self.exec_net:
            config = {}
            if self.device == 'CPU':
                config={'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO'}
            self.exec_net = self.ie.load_network(self.net, self.device,
                                                 config=config, num_requests=0)

        # List that maps infer requests to index of processed chunk from batch.
        # -1 means that request has not been started yet.
        infer_request_input_id = [-1] * len(self.exec_net.requests)
        out_shape = self.net.outputs['output'].shape
        # CTC network produces 1xWxNxC
        output = np.zeros([out_shape[-3], batch_size, out_shape[-1]], dtype=np.float32)

        for inp_id in range(batch_size):
            # Get idle infer request
            infer_request_id = self.exec_net.get_idle_request_id()
            if infer_request_id < 0:
                status = self.exec_net.wait(num_requests=1)
                if status != StatusCode.OK:
                    raise Exception("Wait for idle request failed!")
                infer_request_id = self.exec_net.get_idle_request_id()
                if infer_request_id < 0:
                    raise Exception("Invalid request id!")

            out_id = infer_request_input_id[infer_request_id]
            request = self.exec_net.requests[infer_request_id]

            # Copy output prediction
            if out_id != -1:
                output[:,out_id:out_id+1] = request.output_blobs['output'].buffer

            # Start this request on new data
            infer_request_input_id[infer_request_id] = inp_id
            request.async_infer({'input': data[inp_id]})
            inp_id += 1

        # Wait for the rest of requests
        status = self.exec_net.wait()
        if status != StatusCode.OK:
            raise Exception("Wait for idle request failed!")
        for infer_request_id, out_id in enumerate(infer_request_input_id):
            if out_id == -1:
                continue
            request = self.exec_net.requests[infer_request_id]
            output[:,out_id:out_id+1] = request.output_blobs['output'].buffer

        output = torch.tensor(output)
        if encoder:
            return output
        return self.model.global_norm(output) if self.is_crf else output


    def decode(self, x, beamsize=5, threshold=1e-3, qscores=False, return_path=False):
        if self.is_crf:
            return self.model.decode(x)
        else:
            return self.model.decode(x, beamsize=beamsize, threshold=threshold,
                                     qscores=qscores, return_path=return_path)
