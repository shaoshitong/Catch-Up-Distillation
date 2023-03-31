python train_online_slim_reverse_img_ddp.py  --N 16 --gpu 0,1 \
      --dir ./runs/cifar10-onlineslim-01-beta20/ --weight_prior 20 \
      --learning_rate 2e-4 --dataset cifar10 --warmup_steps 5000 \
      --optimizer adam --batchsize 128 --iterations 500000 \
      --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --loss_type lpips \
      --adaptive_weight True

python train_online_slim_reverse_img_ddp.py  --N 16 --gpu 2,3 \
      --dir ./runs/cifar10-onlineslim-23-beta20/ --weight_prior 20 \
      --learning_rate 2e-4 --dataset cifar10 --warmup_steps 5000 \
      --optimizer adam --batchsize 128 --iterations 500000 \
      --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --loss_type lpips \
      --adaptive_weight False

python train_online_slim_reverse_img_ddp.py  --N 16 --gpu 4,5 \
      --dir ./runs/cifar10-onlineslim-45-beta20/ --weight_prior 20 \
      --learning_rate 2e-4 --dataset cifar10 --warmup_steps 5000 \
      --optimizer adam --batchsize 128 --iterations 500000 \
      --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --loss_type mse \
      --adaptive_weight True


python train_online_slim_reverse_img_ddp.py  --N 16 --gpu 6,7 \
      --dir ./runs/cifar10-onlineslim-67-beta20/ --weight_prior 20 \
      --learning_rate 2e-4 --dataset cifar10 --warmup_steps 5000 \
      --optimizer adam --batchsize 128 --iterations 500000 \
      --config_en configs/cifar10_en.json --config_de configs/cifar10_de.json --loss_type mse \
      --adaptive_weight False