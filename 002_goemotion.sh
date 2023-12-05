#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python 002_main.py --skip_wandb --exp_id 001 --run_id 001 --feats_type sen_trans --model_type ffnn --class_dim 512 --num_train_epochs 20 --train_batch_size 512 --learning_rate 1e-4 --weight_decay 1e-3 --dropout 0.6
