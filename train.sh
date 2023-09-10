#!/bin/bash

OPTION=options/train/ESRGAN/ESRGAN.yml 
export CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES=0 python train.py -opt $OPTION