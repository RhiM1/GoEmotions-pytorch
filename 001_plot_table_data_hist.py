import argparse
import json
import logging
import os
import glob
import wandb
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from attrdict import AttrDict
import gensim.downloader as api

# from transformers import BertConfig, BertTokenizer, AdamW, \
#     get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from sentence_transformers import SentenceTransformer, util

from model import minerva, minerva_ffnn3, minerva_ffnn4, ffnn_init

from utils import init_logger, set_seed, compute_metrics
from data_loader import load_and_cache_examples, GoEmotionsProcessor
from w2v_dataset import get_goem_dataset, get_stratified_ex_idx
from LSA_process import get_lsa_dict

from sklearn.metrics import roc_auc_score

from constants import DATAROOT_group, DATAROOT_ekman, DATAROOT_original

logger = logging.getLogger(__name__)


def count_parameters(model): 
    params = sum(p.numel() for p in model.parameters())
    learned_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return learned_params, params - learned_params

def get_emotions_from_coding(coding):
    
    classes = ['ambiguous', 'neutral', 'negative', 'positive']
    str_emotions = ""
    for i in range(3, -1, -1):
        if coding / 2**i >= 1:
            str_emotions += f"{classes[i]}/"
            coding = coding % 2**i
        # print(f"{i}, {2**i}, {coding}, {str_emotions}")
    str_emotions = str_emotions[0:-1]
    return str_emotions


def main(args):

    args.feats_type

    
    emotion_name = [
        'ambig', 'neutral', '-ve', '+ve',
        'ambig/neutral', 'ambig/-ve', 'ambig/+ve',
        'neutral/-ve', 'neutral/+ve', '-ve/+ve',
        'ambig/neutral/-ve', 'ambig/neutral/+ve',
        'ambig/-ve/+ve', 'neutral/-ve/+ve',
        'ambig/neutral/-ve/+ve'
    ]

    # print(f"feats_types: {feats_types}")
    # print(f"args.feats_type: {args.feats_type}")

    # print(get_emotions_from_coding(15))
    # quit()

    if args.feats_type == 'lsa':
        feats_model = get_lsa_dict('data/LSA/TASA.rda')
    elif args.feats_type == "word2vec":
        feats_model = api.load(args.model_name_or_path)
    elif args.feats_type == "sen_trans":
        feats_model = SentenceTransformer(args.model_name_or_path)

    # Load dataset
    train_dataset = get_goem_dataset(args, mode = 'train', model = feats_model) if args.train_file else None
    dev_dataset = get_goem_dataset(args, mode = 'dev', model = feats_model) if args.dev_file else None
    test_dataset = get_goem_dataset(args, mode = 'test', model = feats_model) if args.test_file else None

    # print(f"test classes: {test_dataset[torch.arange(len(test_dataset))][1].sum(dim = 0)}")

    example_feats, example_classes = test_dataset[0]
    # print(f"features size: {example_feats.size(0)}, num_classes: {example_classes.size(0)}")
    args.input_dim = example_feats.size(0)
    args.num_labels = example_classes.size(0)
    
    emotions_train = torch.stack([train_dataset[i][1] for i in range(len(train_dataset))])
    emotions_dev = torch.stack([dev_dataset[i][1] for i in range(len(dev_dataset))])
    emotions_test = torch.stack([test_dataset[i][1] for i in range(len(test_dataset))])

    multi = torch.tensor([1, 2, 4, 8], dtype = torch.float)
    emotions_train_code = emotions_train @ multi
    emotions_dev_code = emotions_dev @ multi
    emotions_test_code = emotions_test @ multi
    # print(emotions_train)
    # print(emotions_dev)
    # print(emotions_test)

    # uniques_train, counts_train = torch.unique(emotions_train_code, sorted = True, return_counts = True)
    # uniques_dev, counts_dev = torch.unique(emotions_dev_code, sorted = True, return_counts = True)
    # uniques_test, counts_test = torch.unique(emotions_test_code, sorted = True, return_counts = True)
    classes = ['ambiguous', 'neutral', 'negative', 'positive']
    coding = torch.tensor([0, 0, 0, 0])

    train_counts = []
    dev_counts = []
    test_counts = []

    for i in range(1, 16):
        train_counts.append((emotions_train_code == i).type(torch.float).sum().item())
        dev_counts.append((emotions_dev_code == i).type(torch.float).sum().item())
        test_counts.append((emotions_test_code == i).type(torch.float).sum().item())
        # print(f"{i} \t {train_counts}")

    train_counts = torch.tensor(train_counts)
    dev_counts = torch.tensor(dev_counts)
    test_counts = torch.tensor(test_counts)

    # for i in range(len(uniques_train)):
    #     print(f"{i} \t {uniques_train[i]} \t {uniques_dev[i]} \t {uniques_test[i]} \t {counts_train[i]} \t {counts_dev[i]} \t {counts_test[i]} \t {get_emotions_from_coding(uniques_train[i])}")

    # counts_train = counts_train / len(emotions_train_code)   
    # counts_dev = counts_dev / len(emotions_dev_code) 
    # counts_test = counts_test / len(emotions_test_code)  

    train_counts = train_counts / len(emotions_train_code)   
    dev_counts = dev_counts / len(emotions_dev_code) 
    test_counts = test_counts / len(emotions_test_code)  

    ordering = [0, 1, 3, 7, 2, 4, 8, 5, 9, 11, 6, 10, 12, 13, 14]

    strWrite = f"code emotion trainCount devCount testCount\n"
    for i in range(15):
        strWrite += f"{i} {emotion_name[i]} {train_counts[ordering[i]].item()} {dev_counts[ordering[i]].item()} {test_counts[ordering[i]].item()}\n"
        # print(f"{ordering[i] + 1} \t {train_counts[ordering[i]].item()} \t {dev_counts[ordering[i]].item()} \t {test_counts[ordering[i]].item()} \t {get_emotions_from_coding(ordering[i]+1)} \t {emotion_name[i]}")

    print(strWrite)
    with open('tables/GoEmotionHist.txt', 'w') as f:
        f.write(strWrite)
    # print(emotions_train.sum(dim = 0))
    # print(emotions_dev.sum(dim = 0))
    # print(emotions_test.sum(dim = 0))

    quit()
    i = 0
    strWrite = f"i class"
    for j in range(len(corrects)):
        title = args.titles[j] if args.titles is not None else f"model{j}"
        strWrite += f" {title}"
    strWrite += f"\n"
    for classID in range(len(classes)):
        strWrite += f"{i} {classes[classID]}"
        for j in range(len(corrects)):
            strWrite += f" {corrects[j][classID].item() / len(predictions[j])}"
            # num_phone = (labels[j] == phoneID).type(torch.float).sum().item()
            # num_phone_correct = corrects[j][labels[j] == phoneID].type(torch.float).sum().item()
            # prop_phone_correct = num_phone_correct / num_phone
            # strWrite += f" {prop_phone_correct}"
            # print(f"{phoneID}\t{j}\t{prop_phone_correct}")
        strWrite += f"\n"
        i += 1
        # print(f"{phoneID}\t {ID_to_phone[phoneID]}\t {num_phone}\t {num_phone_correct:>.0f}\t {prop_phone_correct:>.4f} ")
    print(strWrite)
    # with open(f'tables/{args.run_id}-timit-phoneAcc.txt', 'w') as f:
    #     f.write(strWrite)



if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()

    # General args
    arg_parser.add_argument(
        "--exp_id", help = "experiment id", default = "test"
    )
    arg_parser.add_argument(
        "--run_id", help = "ID of current run", default = "test"
    )
    arg_parser.add_argument(
        "--pretrained", help = "pretrained model name", default = None, nargs='+'
    )
    arg_parser.add_argument(
        "--seed", help = "only used for BERT", default = 42, type = int
    )
    arg_parser.add_argument(
        "--ckpt_dir", help = "only used for BERT", default = "ckpt/thesis"
    )
    arg_parser.add_argument(
        "--titles", help = "pretrained model name", default = None, nargs='+'
    )


    # Feats args
    arg_parser.add_argument(
        "--taxonomy", help = "one of group, ekman, original", default = "group"
    )
    arg_parser.add_argument(
        "--feats_type", help = "feat extraction model: lsa, word2vec, sen_trans, glove", default = "sen_trans"#, nargs='+'
    )
    arg_parser.add_argument(
        "--model_name_or_path", help = "", default = " "
    )

    # Model args
    arg_parser.add_argument(
        "--model_type", help = "ffnn, minerva, minerva_detEx", default = None, nargs='+'
    )
    arg_parser.add_argument(
        "--eval_batch_size", help = "", default = 32, type = int
    )
    arg_parser.add_argument(
        "--no_cuda", help="", default=False, action='store_true'
    )
    arg_parser.add_argument(
        "--auc_av", help="ROC AUC average method: micro, macro, weighted, samples", default='weighted'
    )


    args = arg_parser.parse_args()


    if args.model_type == 'minerva' or args.model_type == 'minerva_ffnn':
        args.exemplar = True
    else:
        args.exemplar = False

    if args.taxonomy == "group":
        args.data_dir = DATAROOT_group
        args.num_classes = 4
    elif args.taxonomy == "ekman":
        args.data_dir = DATAROOT_ekman
        args.num_classes = 7
    elif args.taxonomy == "original":
        args.data_dir = DATAROOT_original
        args.num_classes = 28
    args.train_file = "train.tsv"
    args.dev_file = "dev.tsv"
    args.test_file = "test.tsv"
    args.label_file = "labels.txt"
    args.threshold = [i/40 for i in range(40)]
    # args.model_name_or_path = []
    # for feats_type in args.feats_type:
    if args.feats_type == 'sen_trans':
        args.model_name_or_path = 'paraphrase-mpnet-base-v2'
    elif args.feats_type == 'word2vec':
        args.model_name_or_path = "word2vec-google-news-300"
    else:
        args.model_name_or_path = " "
    # if args.lr_ex == None:
    #     args.lr_ex = args.learning_rate
    # if args.lr_cr is None:
    #     args.lr_cr = args.lr_ex
        
    # if args.wd_ex is None:
    #     args.wd_ex = args.weight_decay
    # if args.wd_cr is None:
    #     args.wd_cr = args.wd_ex

    args.task = "goemotions"

    # output_dirs = []
    # for pretrained in args.pretrained:
    #     output_dirs.append(f"{args.ckpt_dir}/{pretrained}/checkpoint")
    # args.output_dir = output_dirs
    
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    main(args)
