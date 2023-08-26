from traiNNer.archs.dat_arch import DAT
#from traiNNer.archs.srformer_arch import SRFormer
#from traiNNer.archs.hat_arch import HAT
#from traiNNer.archs.omnisr_arch import OmniSRNet, OSAG
import torch

print("Converting to onnx")

# ! Be sure to check these parameters manually. It needs to be the same you used for trianing.

model = DAT(
  type=DAT,
  upscale=4,
  in_chans=3,
  img_size=64,
  img_range=1.,
  split_size=[8,32],
  depth=[6,6,6,6,6,6],
  embed_dim=180,
  num_heads=[6,6,6,6,6,6],
  expansion_factor=4,
  resi_connection='1conv'
)

#SRFormer light
#model = SRFormer(type=SRFormer, upscale=1, in_chans=3, img_size=64, window_size=16, img_range=1., embed_dim=60, mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv', depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6])

# SRFormer
#model = SRFormer(type=SRFormer, upscale=4, in_chans=3, img_size=48, window_size=22, img_range=1., embed_dim=180, mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv', depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6])

#HAT
#model = HAT(
#    type=HAT, upscale=4, in_chans=3, img_size=64, window_size=16, compress_ratio=24, squeeze_factor=24, conv_scale=0.01, overlap_ratio=0.5, img_range=1., embed_dim=144, mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv', depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6]
#)

#HAT-L
#model = HAT(
#type=HAT,
#upscale=4,
#in_chans=3,
#img_size=64,
#window_size=16,
#compress_ratio=3,
#squeeze_factor=30,
#conv_scale=0.01,
#overlap_ratio=0.5,
#img_range=1.,
#depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
#embed_dim=180,
#num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
#mlp_ratio=2,
#upsampler='pixelshuffle',
#resi_connection='1conv',
#)


# replace with the path to the pth file
state_dict = torch.load("4xNomos8kDAT_net_g_110000.pth")

if "params_ema" in state_dict.keys():
    model.load_state_dict(state_dict["params_ema"])
else:
    model.load_state_dict(state_dict)

model.eval().cuda()

dynamic_axes = {
    "input": {0: "batch_size", 2: "width", 3: "height"},
    "output": {0: "batch_size", 2: "width", 3: "height"},
}

#dummy_input = torch.rand(1, 3, 64, 64).cuda()
#dummy_input = torch.rand(1, 3, 32, 32).cuda()
#dummy_input = torch.rand(1, 3, 16, 16).cuda()
dummy_input = torch.rand(1, 3, 8, 8).cuda()

# fp32 conversion
torch.onnx.export(
    model,
    dummy_input,
    "4xNomos8kDAT_16_fp32.onnx",
    opset_version=14,
    verbose=False,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes=dynamic_axes,
)

# do fp16 after with convertfp32tofp16.py script, after having done onnxsim on the fp32 output
print("Finished FP32 onnx conversion. Now verify, and then run onnxsim")
