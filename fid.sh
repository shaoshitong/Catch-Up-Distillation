

CUDA_VISIBLE_DEVICES=0 python fid.py calc_multiple \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_4/trajs_0 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_4/trajs_1 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_4/trajs_2 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_4/trajs_3 \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz


CUDA_VISIBLE_DEVICES=1 python fid.py calc_multiple \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_4/trajs_0 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_4/trajs_1 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_4/trajs_2 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_4/trajs_3 \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz


CUDA_VISIBLE_DEVICES=2 python fid.py calc_multiple \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_2/trajs_0 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_2/trajs_1 \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

CUDA_VISIBLE_DEVICES=3 python fid.py calc_multiple \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_2/trajs_0 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_2/trajs_1 \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz


CUDA_VISIBLE_DEVICES=4 python fid.py calc_multiple \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_8/trajs_0 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_8/trajs_1 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_8/trajs_2 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_8/trajs_3 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_8/trajs_4 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_8/trajs_5 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_8/trajs_6 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_8/trajs_7 \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

CUDA_VISIBLE_DEVICES=5 python fid.py calc_multiple \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_8/trajs_0 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_8/trajs_1 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_8/trajs_2 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_8/trajs_3 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_8/trajs_4 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_8/trajs_5 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_8/trajs_6 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_8/trajs_7 \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz


CUDA_VISIBLE_DEVICES=6 python fid.py calc_multiple \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_16/trajs_0 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_16/trajs_1 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_16/trajs_2 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_16/trajs_3 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_16/trajs_4 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_16/trajs_5 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_16/trajs_6 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_16/trajs_7 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_16/trajs_8 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_16/trajs_9 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_16/trajs_10 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_16/trajs_11 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_16/trajs_12 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_16/trajs_13 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_16/trajs_14 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_16/trajs_15 \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

CUDA_VISIBLE_DEVICES=7 python fid.py calc_multiple \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_16/trajs_0 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_16/trajs_1 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_16/trajs_2 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_16/trajs_3 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_16/trajs_4 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_16/trajs_5 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_16/trajs_6 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_16/trajs_7 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_16/trajs_8 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_16/trajs_9 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_16/trajs_10 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_16/trajs_11 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_16/trajs_12 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_16/trajs_13 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_16/trajs_14 \
    --images=./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_16/trajs_15 \
    --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz

