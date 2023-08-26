import onnx

onnx_model = onnx.load("4xNomos8kSCHAT-L_131616.onnx")
onnx.checker.check_model(onnx_model)
