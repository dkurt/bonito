import os
import torch
import argparse
from bonito.util import load_model

parser = argparse.ArgumentParser(description='Use this script to prepare OpenVINO IR for trained model')
parser.add_argument("model_directory")
parser.add_argument("--half", action="store_true", default=False)
parser.add_argument("--weights", default="0", type=str)
args = parser.parse_args()

__dir__ = os.path.dirname(os.path.dirname(__file__))
__models__ = os.path.join(__dir__, "models")
dirname = args.model_directory
if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models__, dirname)):
    dirname = os.path.join(__models__, dirname)

model = load_model(dirname, 'cpu', weights=int(args.weights), half=args.half)


# Convert to ONNX
onnx_path = os.path.join(dirname, model.config['model']) + '.onnx'
inp = torch.randn(1, 1, 1000)  # Just dummy input shape. We will reshape model later
model.eval()
with torch.no_grad():
    torch.onnx.export(model, inp, onnx_path,
                    input_names=['input'],
                    output_names=['output'],
                    operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)


# Convert to IR
import mo_onnx
import subprocess
model_name = model.config['model'] + ('_fp16' if args.half else '')
subprocess.call([mo_onnx.__file__,
                 '--input_model', onnx_path,
                 '--extension', os.path.join(os.path.dirname(__file__), 'mo_extension'),
                 '--keep_shape_ops',
                 '--model_name', model_name,
                 '--data_type', 'FP16' if args.half else 'FP32',
                 '--input_shape=[1,1,1,1000]',
                 '--output_dir', dirname])
