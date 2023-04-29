#!/bin/bash

for((i=0;i<=15;i++));  
do   
CUDA_VISIBLE_DEVICES=1 python fid.py calc \
    --images=runs/cifar10-onlineslim-predstep-1-3-beta20/test_heun_2/trajs_${i} \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
done 

"""
CUDA_VISIBLE_DEVICES=1 python fid.py calc \
    --images=./runs/cifar10-onlineslim-predstep-2-uniform-shakedrop-sam-beta20/test_4/ \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

CUDA_VISIBLE_DEVICES=2 python fid.py calc \
    --images=./runs/cifar10-onlineslim-predstep-2-uniform-shakedrop-beta20/test_4/ \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
"""