#!/bin/bash

cd ./runs/cifar10-onlineslim-predstep-1-uniform-shakedrop0.75-sam-t-beta20
rm -rf ./flow_model_0_ema.pth
rm -rf ./flow_model_50000_ema.pth
rm -rf ./flow_model_100000_ema.pth
rm -rf ./flow_model_150000_ema.pth
rm -rf ./flow_model_200000_ema.pth
rm -rf ./flow_model_250000_ema.pth
rm -rf ./flow_model_300000_ema.pth
rm -rf ./flow_model_350000_ema.pth
rm -rf ./flow_model_400000_ema.pth
rm -rf ./flow_model_450000_ema.pth

rm -rf ./forward_model_0_ema.pth
rm -rf ./forward_model_50000_ema.pth
rm -rf ./forward_model_100000_ema.pth
rm -rf ./forward_model_150000_ema.pth
rm -rf ./forward_model_200000_ema.pth
rm -rf ./forward_model_250000_ema.pth
rm -rf ./forward_model_300000_ema.pth
rm -rf ./forward_model_350000_ema.pth
rm -rf ./forward_model_400000_ema.pth
rm -rf ./forward_model_450000_ema.pth

rm -rf training_state_0.pth
rm -rf training_state_50000.pth
rm -rf training_state_150000.pth
rm -rf training_state_200000.pth
rm -rf training_state_250000.pth
rm -rf training_state_300000.pth
rm -rf training_state_350000.pth
rm -rf training_state_400000.pth
rm -rf training_state_450000.pth
rm -rf training_state_500000.pth
rm -rf *.jpg
rm -rf test_*
rm -rf events.out*