#!/bin/bash

for((i=1;i>=0;i--));  
do   
CUDA_VISIBLE_DEVICES=0 python fid.py calc \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-sam-beta20/test_2/trajs_${i} \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
done

for((i=1;i>=0;i--));  
do   
CUDA_VISIBLE_DEVICES=1 python fid.py calc \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-sam-t-beta20/test_2/trajs_${i} \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
done

for((i=1;i>=0;i--));  
do   
CUDA_VISIBLE_DEVICES=2 python fid.py calc \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-sam-beta20/test_heun_2/trajs_${i} \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
done

for((i=1;i>=0;i--));  
do   
CUDA_VISIBLE_DEVICES=3 python fid.py calc \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-sam-t-beta20/test_heun_2/trajs_${i} \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
done