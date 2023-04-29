
python generate.py --gpu 0 --dir ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-sam-beta20/test_8 --N 8 --res 32 \
      --input_nc 3 --num_samples 50000 --ckpt ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-sam-beta20/flow_model_500000_ema.pth \
      --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --dataset cifar10 --save_sub_traj --shakedrop --phi 0.75

python generate.py --gpu 1 --dir ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-sam-beta20/test_heun_8/ --N 15 --res 32 \
      --input_nc 3 --num_samples 50000 --ckpt ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-sam-beta20/flow_model_500000_ema.pth \
      --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --dataset cifar10 --save_sub_traj --shakedrop --phi 0.75 --solver heun

python generate.py --gpu 2 --dir ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-sam-beta20/test_2/ --N 2 --res 32 \
      --input_nc 3 --num_samples 50000 --ckpt ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-sam-beta20/flow_model_500000_ema.pth \
      --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --dataset cifar10 --save_sub_traj --shakedrop --phi 0.75

python generate.py --gpu 3 --dir ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-sam-beta20/test_heun_2/ --N 3 --res 32 \
      --input_nc 3 --num_samples 50000 --ckpt ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-sam-beta20/flow_model_500000_ema.pth \
      --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --dataset cifar10 --save_sub_traj --shakedrop --phi 0.75 --solver heun


====================================================


python generate.py --gpu 4 --dir ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-sam-t-beta20/test_8 --N 8 --res 32 \
      --input_nc 3 --num_samples 50000 --ckpt ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-sam-t-beta20/flow_model_500000_ema.pth \
      --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --dataset cifar10 --save_sub_traj --shakedrop --phi 0.75

python generate.py --gpu 5 --dir ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-sam-t-beta20/test_heun_8/ --N 15 --res 32 \
      --input_nc 3 --num_samples 50000 --ckpt ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-sam-t-beta20/flow_model_500000_ema.pth \
      --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --dataset cifar10 --save_sub_traj --shakedrop --phi 0.75 --solver heun

python generate.py --gpu 6 --dir ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-sam-t-beta20/test_2/ --N 2 --res 32 \
      --input_nc 3 --num_samples 50000 --ckpt ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-sam-t-beta20/flow_model_500000_ema.pth \
      --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --dataset cifar10 --save_sub_traj --shakedrop --phi 0.75

python generate.py --gpu 7 --dir ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-sam-t-beta20/test_heun_2/ --N 3 --res 32 \
      --input_nc 3 --num_samples 50000 --ckpt ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-sam-t-beta20/flow_model_500000_ema.pth \
      --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --dataset cifar10 --save_sub_traj --shakedrop --phi 0.75 --solver heun
