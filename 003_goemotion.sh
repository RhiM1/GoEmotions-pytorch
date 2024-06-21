#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python 001_csl_main.py --exp_id 000 --run_id 001 --skip_wandb --feats_type lsa --model_type minerva_ffnn --p_factor 1 --num_ex 8192 --fix_ex --skip_train
CUDA_VISIBLE_DEVICES=0 python 001_csl_main.py --exp_id 000 --run_id 001 --skip_wandb --feats_type word2vec --model_type minerva_ffnn --p_factor 1 --num_ex 8192 --skip_train
CUDA_VISIBLE_DEVICES=0 python 001_csl_main.py --exp_id 000 --run_id 001 --skip_wandb --feats_type sen_trans --model_type minerva_ffnn --p_factor 1 --num_ex 8192 --skip_train
CUDA_VISIBLE_DEVICES=0 python 001_csl_main.py --exp_id 001 --run_id 001 --skip_wandb --num_train_epochs 1 --feats_type lsa --model_type ffnn_init --feat_dim 1024 --class_dim 1024 --train_batch_size 512 --learning_rate 1e-4 --weight_decay 1e-1 --do_class 0.3
CUDA_VISIBLE_DEVICES=0 python 001_csl_main.py --exp_id 002 --run_id 001 --skip_wandb --num_train_epochs 1 --feats_type lsa --model_type minerva_ffnn --use_thresh --use_mult --feat_dim 32 --train_batch_size 512 --learning_rate 1e-2 --weight_decay 1e-2 --p_factor 3 --do_feat 0 --do_class 0 --num_ex 8192
CUDA_VISIBLE_DEVICES=0 python 001_csl_main.py --exp_id 003 --run_id 001 --skip_wandb --num_train_epochs 1 --feats_type lsa --model_type minerva_ffnn --use_thresh --use_mult --feat_dim 32 --train_batch_size 512 --learning_rate 1e-2 --weight_decay 1e-2 --lr_ex 1e-3 --wd_ex 0 --p_factor 3 --do_feat 0.2 --do_class 0 --num_ex 8192 --train_ex_class


