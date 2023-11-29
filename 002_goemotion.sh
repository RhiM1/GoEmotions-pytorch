#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python 002_main.py --exp_id 001 --run_id test --feats_type lsa --model_type ffnn --class_dim 512 --num_train_epochs 10 --train_batch_size 16 --weight_decay 0 --learning_rate 1e-5 --dropout 0
