from onnx2torch import convert
from imports import CNV


def load_model(onnx_model):
    # Convert ONNX model to PyTorch
    torch_model = convert(onnx_model)
    print(torch_model)
