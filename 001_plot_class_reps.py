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
import matplotlib.pyplot as plt
import seaborn as sb

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


def plot_confusion_heatmap_1col(class_reps, num_classes = 4, titles = None, share_color_bar = True):
    """Plot confusion matrix using heatmap.
 
    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.
 
    """

    classes = ['ambiguous', 'neutral', 'negative', 'positive']

    noCols = len(class_reps) if len(class_reps) < 3 else 3
    noRows = len(class_reps) // noCols
    noRows = noRows + 1 if len(class_reps) % noCols > 0 else noRows
    # widths = [1,0.1]
    # widths.append(0.1)
    # heights = [1] * (noRows)
    # heights.append(0.1)

    minHeat = class_reps.min()
    maxHeat = class_reps.max()

    fig, axs = plt.subplots(1, noCols, sharex = True, sharey = True)
    if share_color_bar:
        cbar_ax = fig.add_axes([.85, 0.25, .05, 0.4])
        fig.set_figwidth(2 * noCols + 1)
    else:
        fig.set_figwidth(2.5)

    # print(f"axs:{axs}")
    # print(f"axs[0] length:{len(axs[0])}")
    # if len(axs) == 1:
    if noRows == 1:
        axs = [axs]
    print(f"rows: {noRows}, cols: {noCols}")
    print(f"axes length: {len(axs)}")
    if noCols == 1:
        axs = [axs]

    for i in range(len(class_reps)):
        class_rep = class_reps[i]
        # confMat = confMat[:, labelOrder]
        # confMat = confMat[predsOrder]

        print(f"{i}\n{class_rep}")
        g = sb.heatmap(
            class_rep.detach().numpy(), 
            annot=False, 
            cmap =  "RdBu_r", #sb.color_palette("RdBl", as_cmap=True), 
            # cbar_kws={'orientation': 'horizontal', 'label': 'Proportion of true class', 'shrink': 1}, 
            cbar = True if i == 0 else False,
            # cbar = False,
            cbar_ax = cbar_ax if share_color_bar else None,
            square = True,
            xticklabels = classes,# if row == noRows - 1 else [],
            yticklabels = classes,# if col == 0 else [],
            ax = axs[i // noCols][i % noCols],
            vmin = minHeat.detach().numpy(),
            vmax = maxHeat.detach().numpy(),
            center = 0
        )
        
        # for tick in axs[row][0].get_xticklabels():
        #     tick.set_fontsize(7)
        
        # for tick in axs[row][0].get_yticklabels():
        #     tick.set_fontsize(7)

        # if col == 0:
        # g.set(ylabel="Predicted class")
        # if i == noRows - 1:
        #     g.set(xlabel="True class")

        if titles is not None:
            axs[i // noCols][i % noCols].title.set_text(titles[i])



    if share_color_bar:
        fig.tight_layout(rect=[0, 0, .85, 1])
    else:
        fig.tight_layout()
    plt.savefig(f"figures/{args.run_id}-timit-classreps.eps", format = 'eps')
    plt.show()


def main(args):

    logger.info("Training/evaluation parameters {}".format(args))
    init_logger()
    set_seed(args)

    model_types = args.model_type
    pretrained = args.pretrained
    output_dirs = args.output_dir

    class_reps = []
    class_reps_corr = []
    class_reps_cov = []

    for i in range(len(model_types)):
        args.model_type = model_types[i]
        args.pretrained = pretrained[i]
        args.output_dir = output_dirs[i]

        if args.model_type == 'minerva_ffnn4':
            model = minerva_ffnn4(args, load_dir = args.output_dir)
        elif args.model_type == 'minerva_ffnn3':
            model = minerva_ffnn3(args, load_dir = args.output_dir)
        elif args.model_type == 'ffnn_init':
            model = ffnn_init(args, load_dir = args.output_dir)

        print(model)
        # print(model.state_dict())

        class_reps.append(model.class_reps.to('cpu'))
        class_reps_cov.append(class_reps[-1] @ class_reps[-1].t())
        class_reps_corr.append(
            torch.nn.functional.normalize(class_reps[-1], dim = 1) @ torch.nn.functional.normalize(class_reps[-1], dim = 1).t()
        )
    
    class_reps = torch.stack(class_reps)
    class_reps_cov = torch.stack(class_reps_cov)
    class_reps_corr = torch.stack(class_reps_corr)

    print(class_reps)
    print(class_reps_cov)
    print(class_reps_corr)

    # plot_confusion_heatmap_1col(class_reps)
    # plot_confusion_heatmap_1col(class_reps_cov)
    plot_confusion_heatmap_1col(class_reps_corr, titles = args.titles)
    


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
    # arg_parser.add_argument(
    #     "--feats_type", help = "feat extraction model: lsa, word2vec, sen_trans, glove", default = "sen_trans", nargs='+'
    # )
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
    args.model_name_or_path = []
    # for feats_type in args.feats_type:
    #     if feats_type == 'sen_trans':
    #         args.model_name_or_path.append('paraphrase-mpnet-base-v2')
    #     elif args.feats_type == 'word2vec':
    #         args.model_name_or_path.append()
    #     else:
    #         args.model_name_or_path.append(" ")
    # if args.lr_ex == None:
    #     args.lr_ex = args.learning_rate
    # if args.lr_cr is None:
    #     args.lr_cr = args.lr_ex
        
    # if args.wd_ex is None:
    #     args.wd_ex = args.weight_decay
    # if args.wd_cr is None:
    #     args.wd_cr = args.wd_ex

    args.task = "goemotions"

    output_dirs = []
    for pretrained in args.pretrained:
        output_dirs.append(f"{args.ckpt_dir}/{pretrained}/checkpoint")
    args.output_dir = output_dirs
    
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    main(args)
