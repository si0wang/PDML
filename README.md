## Overview
Code of ICML 2023 paper: "Live in the Moment: Learning Dynamics Model Adapted to Evolving Policy"

## Usage

PDML-MBPO
> CUDA_VISIBLE_DEVICES=0 python main_pdml.py --env_name 'Humanoid-v2' --num_epoch 400 --exp_name humanoid_pdml --seed 4 --reweight_model TV --reweight_rollout TV


## code structure
During training, 'exp' folder will created aside 'PDML' folder.

## Dependencies
MuJoCo 1.5 & MuJoCo 2.0

## Reference
This code is built on a pytorch implementation MBPO: https://github.com/Xingyu-Lin/mbpo_pytorch
