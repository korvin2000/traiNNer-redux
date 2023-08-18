import onnx
from onnxconverter_common import float16

model = onnx.load("2xHFA2kAVCSRFormer_light_64.onnx")
model_fp16 = float16.convertfp32tofp16(model)
onnx.save(model_fp16,"2xHFA2kAVCSRFormer_light_64_fp16.onnx")
