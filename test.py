import onnx
from quantize import quantize,QuantizationMode

# Load the onnx model
model = onnx.load('mobilenet.onnx')
# quantize
quantizated_model = quantize(model,quantization_mode=QuantizationMode.IntegerOps)
#Save the quantized model
onnx.save(quantizated_model,'mobilenet_q.onnx')
