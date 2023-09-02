(Fork of BasicSR joeyballentine/traiNNer-redux) Open Source Image and Video Restoration Toolbox for Super-resolution, Denoise, Deblurring, etc. Currently, it includes EDSR, RCAN, SRResNet, SRGAN, ESRGAN, EDVR, BasicVSR, SwinIR, ECBSR, OmniSR, HAT, GRL, A-ESRGAN, etc. Also support StyleGAN2, DFDNet.

***************************
NEW ADD ARCH SUPPORT
- [ESRGAN 8X]
- [Omnisr](https://github.com/Francis0625/Omni-SR)
  - The arch implementation of Omnisr is from [Omnisr](https://github.com/Francis0625/Omni-SR). The LICENSE of Omnisr is [Apache License 2.0]. The LICENSE is included as [LICENSE_Omnisr](LICENSE/LICENSE_Omnisr).
- [HAT](https://github.com/XPixelGroup/HAT)
  - The arch implementation of HAT is from [HAT](https://github.com/XPixelGroup/HAT). The LICENSE of HAT is [MIT License]. The LICENSE is included as [LICENSE_HAT](LICENSE/LICENSE_HAT).
- [GRL](https://github.com/ofsoundof/GRL-Image-Restoration/tree/main)
  - The arch implementation of GRL is from [GRL](https://github.com/ofsoundof/GRL-Image-Restoration/tree/main). The LICENSE of GRL is [MIT License]. 
- [ESWT](https://github.com/Fried-Rice-Lab/FriedRiceLab)
  - The arch implementation of ESWT is from [ESWT](https://github.com/Fried-Rice-Lab/FriedRiceLab). The LICENSE of ESWT is [MIT License]. 
- [SRFormer](https://github.com/HVision-NKU/SRFormer)
  - The arch implementation of SRFormer is from [SRFormer](https://github.com/HVision-NKU/SRFormer). The LICENSE of SRFormer is [Apache License 2.0]. 
- [A-ESRGAN](https://github.com/stroking-fishes-ml-corp/A-ESRGAN)
  - The arch implementation of A-ESRGAN is from [A-ESRGAN](https://github.com/stroking-fishes-ml-corp/A-ESRGAN). The LICENSE of A-ESRGAN is [BSD 3-Clause "New" or "Revised" License].
- [DAT](https://github.com/zhengchen1999/DAT)
  - The arch implementation of DAT is from [A-ESRGAN](https://github.com/zhengchen1999/DAT). The LICENSE of DAT is [Apache License 2.0]. 
***************************
NEW FEATURE SUPPORT
-  ContextualLoss weight
- amp support
  ```
  try_autoamp_g: True # enable amp Automatic mixed precision for network_g. if loss inf or nan or error just set to False
  try_autoamp_d: True # enable amp Automatic mixed precision for network_d. if loss inf or nan or error just set to False
  ```
  
***************************
TIPS
- PairedImageDataset set high_order_degradation : False
- RealESRGANDataset set high_order_degradation : True
- model_type can use GeneralGANModel or GeneralNetModel for ,ost of archs
- To use Automatic mixed precision, edit the yml
- e.g.  if you want use omnisr-gan with amp, just use model_type: GeneralGANModel and edit the .yml 's following part
```
name: train_OmniSRGANModel_SRx4_scratch_P48W8_DIV2K_500k_B4G8
try_autoamp_g: True # enable amp Automatic mixed precision for network_g. if loss inf or nan or error just set to False
try_autoamp_d: True # enable amp Automatic mixed precision for network_d. if loss inf or nan or error just set to False
network_g:
  type: OmniSRNet
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 64
  upsampling: 4
  res_num: 5
  block_num: 1
  bias: True
  block_script_name: OSA
  block_class_name: OSA_Block
  window_size: 8
  pe: True
  ffn_bias: True
```
- If you want to use aesrgan's network_d for other network_d. you should edit the .yml network_d to multiscale type
- e.g. if you want use omnisr as network_g and multiscale as network_d, just edit the .yml 's following part
```
# network structures

network_g:
  type: OmniSRNet
  num_in_ch: 3
  num_out_ch: 3
  num_feat: 64
  upsampling: 4
  res_num: 5
  block_num: 1
  bias: True
  block_script_name: OSA
  block_class_name: OSA_Block
  window_size: 8
  pe: True
  ffn_bias: True

network_d:
  type: multiscale
  num_in_ch: 3
  num_feat: 64
  num_D: 2
```
***************************
for easy use here are examples for network_g
```
# network structures
network_g:
  # DAT-S, nees to set batch size >1
  type: DAT
  upscale: 4
  in_chans: 3
  img_size: 64
  img_range: 1.
  split_size: [8,16]
  depth: [6,6,6,6,6,6]
  embed_dim: 180
  num_heads: [6,6,6,6,6,6]
  expansion_factor: 2
  resi_connection: '1conv'

```
