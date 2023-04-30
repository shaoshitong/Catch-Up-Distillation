#!/bin/bash

python generate.py --gpu 0 --dir ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_2 --N 2 --res 32 \
      --input_nc 3 --num_samples 50000 --ckpt ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/flow_model_150000_ema.pth \
      --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --dataset cifar10 --save_sub_traj --shakedrop --phi 0.75 --discrete --total_N 16

python generate.py --gpu 1 --dir ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_2/ --N 3 --res 32 \
      --input_nc 3 --num_samples 50000 --ckpt ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/flow_model_150000_ema.pth \
      --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --dataset cifar10 --save_sub_traj --shakedrop --phi 0.75 --solver heun --discrete --total_N 16

python generate.py --gpu 2 --dir ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_4/ --N 4 --res 32 \
      --input_nc 3 --num_samples 50000 --ckpt ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/flow_model_150000_ema.pth \
      --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --dataset cifar10 --save_sub_traj --shakedrop --phi 0.75 --discrete --total_N 16

python generate.py --gpu 3 --dir ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/test_heun_4/ --N 7 --res 32 \
      --input_nc 3 --num_samples 50000 --ckpt ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-discrete-beta20/flow_model_150000_ema.pth \
      --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --dataset cifar10 --save_sub_traj --shakedrop --phi 0.75 --solver heun --discrete --total_N 16


# ==========================================


python generate.py --gpu 0 --dir ./runs/cifar10-onlineslim-predstep-2-uniform-beta20/test_16 --N 16 --res 32 \
      --input_nc 3 --num_samples 50000 --ckpt ./runs/cifar10-onlineslim-predstep-2-uniform-beta20/flow_model_150000_ema.pth \
      --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --dataset cifar10 --save_sub_traj --generator 2 \
      --generator_path ./runs/cifar10-onlineslim-predstep-2-uniform-beta20/generator_list_500000_ema.pth