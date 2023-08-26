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
- To use Automatic mixed precision, edit the yml files in options\train\ESWT\ 
- e.g.  if you want use omnisr-gan with amp, just edit the 4x_trainESWTGAN_SR_scratch-DIV2K.yml 's following part
```
name: train_ESWTGANModel_SRx4_scratch_P48W8_DIV2K_500k_B4G8
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
- If you want to use aesrgan's network_d for other network_d. you should edit the [4x_train_multiaesrgan_plus.yml](https://github.com/FlotingDream/traiNNer-redux-FJ/blob/master/options/train/AESRGAN/4x_train_multiaesrgan_plus.yml)
- e.g. if you want use omnisr as network_g and multiscale as network_d, just edit the 4x_train_multiaesrgan_plus.yml 's following part
```
# network structures
# network_g:
#   type: RRDBNet
#   num_in_ch: 3
#   num_out_ch: 3
#   num_feat: 64
#   num_block: 23
#   num_grow_ch: 32

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

#network_d:
#  type: UNetDiscriminatorAesrgan
#  num_in_ch: 3
#  num_feat: 64
#  skip_connection: True

network_d:
  type: multiscale
  num_in_ch: 3
  num_feat: 64
  num_D: 2
```

![visitors](https://visitor-badge.glitch.me/badge?page_id=XPixelGroup/BasicSR) (start from 2022-11-06)
