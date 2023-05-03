#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python run_goemotions_minerva.py --taxonomy original-minerva-002
CUDA_VISIBLE_DEVICES=0 python run_goemotions_minerva.py --taxonomy original-minerva-004