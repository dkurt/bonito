import os
import numpy as np
import torch

from bonito.nn import Swish

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
        self.is_ctc = model.config['model']['package'] == 'bonito.ctc'
        self.is_crf = model.config['model']['package'] == 'bonito.crf'
        # CRF
        if self.is_crf:
            self.seqdist = self.model.seqdist
            self.encoder = self

        self.ie = IECore()
        self.half = half
        self.dirname = dirname
        self.net = None
        self.exec_net = None


    def eval(self):
        pass


    def half(self):
        return self


    def to(self, device):
        self.device = str(device).upper()


    def load_network(self, chunksize, force=False):
        model_name = 'model' + ('_fp16' if self.half else '')
        xml_path, bin_path = [os.path.join(self.dirname, model_name) + ext for ext in ['.xml', '.bin']]
        if not os.path.exists(xml_path) or not os.path.exists(bin_path) or force:
            # Just a dummy input to make forward pass
            inp = torch.randn([1, 1, chunksize])
            onnx_path = os.path.join(self.dirname, 'model.onnx')

            # There is an issue with Swish at export step so we temporarly use default implementation
            def swish_fake_forward(self, x):
                return x * torch.sigmoid(x)

            swish_origin_forward = Swish.forward
            Swish.forward = swish_fake_forward
            torch.onnx.export(self.model if self.is_ctc else self.model.encoder,
                              inp, onnx_path, input_names=['input'], output_names=['output'],
                              opset_version=11)
            Swish.forward = swish_origin_forward

            import mo_onnx
            if self.is_ctc:
                mo_onnx(input_model=onnx_path,
                        data_type=('FP16' if self.half else 'FP32'),
                        output_dir=self.dirname,
                        extensions=os.path.join(os.path.dirname(__file__), 'mo_extension'),
                        input_shape='[1,1,1,{}]'.format(chunksize))
            else:
                mo_onnx(input_model=onnx_path,
                        data_type=('FP16' if self.half else 'FP32'),
                        output_dir=self.dirname)
            os.remove(onnx_path)

        self.net = self.ie.read_network(xml_path, bin_path)


    def __call__(self, data):
        # Conv1D -> Conv2D efficiency optimization for CTC model
        if self.is_ctc:
            data = np.expand_dims(data, axis=2)  # 1D->2D
        batch_size = data.shape[0]
        inp_shape = list(data.shape)
        inp_shape[0] = 1  # We will run the batch asynchronously

        # Prepare network at first run
        if not self.net:
            self.load_network(inp_shape[-1])

        # Input shape changed - we need to reload network
        if self.net.input_info['input'].tensor_desc.dims != inp_shape:
            # CTC model is reshapeable so it's enough to just call net.reshape.
            if self.is_ctc:
                self.net.reshape({'input': inp_shape})
            else:
                self.load_network(inp_shape[-1], force=True)
            self.exec_net = None

        if not self.exec_net:
            config = {}
            if self.device == 'CPU':
                config={'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO'}
            self.exec_net = self.ie.load_network(self.net, self.device,
                                                 config=config, num_requests=0)

        for out_name in self.net.outputs.keys():
            pass

        # List that maps infer requests to index of processed chunk from batch.
        # -1 means that request has not been started yet.
        infer_request_input_id = [-1] * len(self.exec_net.requests)
        out_shape = self.net.outputs[out_name].shape
        # CTC network produces 1xWxNxC
        # CTF network produces WxNxC (W - spatial, N - batch size, C - features)
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
                output[:,out_id:out_id+1] = request.output_blobs[out_name].buffer

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
            output[:,out_id:out_id+1] = request.output_blobs[out_name].buffer

        output = torch.tensor(output)
        return output if self.is_ctc else self.model.global_norm(output)


    def decode(self, x, beamsize=5, threshold=1e-3, qscores=False, return_path=False):
        return self.model.decode(x, beamsize=beamsize, threshold=threshold,
                                 qscores=qscores, return_path=return_path)
