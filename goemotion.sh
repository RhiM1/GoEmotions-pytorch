#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python run_goemotions_minerva.py --taxonomy group-minerva-012
CUDA_VISIBLE_DEVICES=0 python run_goemotions_minerva.py --taxonomy group-minerva-011