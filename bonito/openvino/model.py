import os
import numpy as np
import torch

try:
    from openvino.inference_engine import IECore, StatusCode
    from .loader import torch2openvino
except ImportError:
    pass

class OpenVINOModel:

    def __init__(self, model, half, dirname):
        self.model = model
        self.alphabet = model.alphabet
        self.parameters = model.parameters
        self.stride = model.stride

        model_name = 'model' + ('_fp16' if half else '')
        xml_path, bin_path = [os.path.join(dirname, model_name) + ext for ext in ['.xml', '.bin']]
        self.ie = IECore()
        if os.path.exists(xml_path) and os.path.exists(bin_path):
            self.net = self.ie.read_network(xml_path, bin_path)
        else:
            self.net = torch2openvino(model)
        self.exec_net = None


    def eval(self):
        pass


    def half(self):
        return self


    def to(self, device):
        self.device = str(device).upper()


    def __call__(self, data):
        data = np.expand_dims(data, axis=2)  # 1D->2D
        batch_size = data.shape[0]
        inp_shape = list(data.shape)
        inp_shape[0] = 1  # We will run the batch asynchronously
        if not self.exec_net or self.exec_net.input_info['input'].tensor_desc.dims != inp_shape:
            self.net.reshape({'input': inp_shape})
            config = {}
            if self.device == 'CPU':
                config={'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO'}
            self.exec_net = self.ie.load_network(self.net, self.device,
                                                 config=config, num_requests=0)

        # List that maps infer requests to index of processed chunk from batch.
        # -1 means that request has not been started yet.
        infer_request_input_id = [-1] * len(self.exec_net.requests)
        output = np.zeros([batch_size] + self.net.outputs['output'].shape[1:], dtype=np.float32)

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
                output[out_id] = request.output_blobs['output'].buffer

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
            output[out_id] = request.output_blobs['output'].buffer

        output = np.squeeze(output, axis=2)  # 2D->1D
        output = output.transpose(1, 0, 2)  # Model should produce WNC (width, batch, features)
        return torch.tensor(output)


    def decode(self, x, beamsize=5, threshold=1e-3, qscores=False, return_path=False):
        return self.model.decode(x, beamsize=beamsize, threshold=threshold,
                                 qscores=qscores, return_path=return_path)