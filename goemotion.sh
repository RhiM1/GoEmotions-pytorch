#!/bin/bash
#CUDA_VISIBLE_DEVICES=0 python 001_main.py --config bert_minerva_001_007 --skip_wandb
#CUDA_VISIBLE_DEVICES=0 python 001_main.py --config bert_minerva_001_test --skip_wandb
#CUDA_VISIBLE_DEVICES=0 python 001_main.py --config bert_ffnn_001_test --skip_wandb
CUDA_VISIBLE_DEVICES=0 python 001_main.py --config bert_minerva_001_test --skip_wandb
