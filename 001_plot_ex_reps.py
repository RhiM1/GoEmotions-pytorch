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

from model import minerva, minerva_ffnn3, minerva_ffnn4, ffnn_init, minerva_ffnn5

from utils import init_logger, set_seed, compute_metrics
from data_loader import load_and_cache_examples, GoEmotionsProcessor
from w2v_dataset import get_goem_dataset, get_stratified_ex_idx
from LSA_process import get_lsa_dict

from sklearn.metrics import roc_auc_score

from constants import DATAROOT_group, DATAROOT_ekman, DATAROOT_original

logger = logging.getLogger(__name__)


def evaluate(args, model, eval_dataset, mode, ex_dataset = None, epoch=None, thresholds = None):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    if args.exemplar and not args.fix_ex:
        total_ex = len(ex_dataset)
        ex_so_far = 0
        ex_dataloader = iter(DataLoader(ex_dataset, sampler=RandomSampler(ex_dataset), batch_size=args.num_ex))

    # Eval!
    if epoch != None:
        logger.info("***** Running evaluation on {} dataset ({} epoch) *****".format(mode, epoch))
    else:
        logger.info("***** Running evaluation on {} dataset *****".format(mode))
    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    m_activation = 0
    ms_activation = 0
    m_echo = 0
    ms_echo = 0
    m_logits = 0
    ms_logits = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            if args.feats_type == 'bert':
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3]
                }
                if args.exemplar and not args.fix_ex:
                    ex_so_far += args.num_ex
                    if ex_so_far > total_ex:
                        ex_dataloader = iter(DataLoader(ex_dataset, sampler=RandomSampler(ex_dataset), batch_size=args.num_ex))
                        ex_so_far = args.num_ex
                    exemplars = next(ex_dataloader)
                    inputs['exemplars'] = [exemplar.to(args.device) for exemplar in exemplars]

            else:
                inputs = {
                    "features": batch[0],
                    "labels": batch[1]
                }
                if args.exemplar and not args.fix_ex:
                    ex_so_far += args.num_ex
                    if ex_so_far > total_ex:
                        ex_dataloader = iter(DataLoader(ex_dataset, sampler=RandomSampler(ex_dataset), batch_size=args.num_ex))
                        ex_so_far = args.num_ex
                    ex_features, ex_classes = next(ex_dataloader)
                    inputs['ex_features'] = ex_features.to(args.device)
                    inputs['ex_classes'] = ex_classes.to(args.device)

            output = model(**inputs)
            tmp_eval_loss = output['loss']
            logits = output['logits']
            a = output['activations']
            m_activation += a.mean().item()
            ms_activation += torch.pow(a, 2).mean().item()
            echo = output['echo']
            m_echo += echo.mean().item()
            ms_echo += torch.pow(echo, 2).mean().item()
            m_logits += logits.mean().item()
            ms_logits += torch.pow(logits, 2).mean().item()

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = 1 / (1 + np.exp(-logits.detach().cpu().numpy()))  # Sigmoid
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, 1 / (1 + np.exp(-logits.detach().cpu().numpy())), axis=0)  # Sigmoid
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    m_activation = m_activation / nb_eval_steps
    ms_activation = ms_activation / nb_eval_steps
    m_echo = m_echo / nb_eval_steps
    ms_echo = ms_echo / nb_eval_steps
    m_logits = m_logits / nb_eval_steps
    ms_logits = ms_logits / nb_eval_steps
    print(f"mean activation: {m_activation}, root mean square activation: {ms_activation**0.5}")
    print(f"mean echo: {m_echo}, root mean square echo: {ms_echo**0.5}")
    print(f"mean logits: {m_logits}, root mean square logits: {ms_logits**0.5}")
    results = {
        str(mode) + "_loss": eval_loss
    }
    
    preds_ = np.copy(preds)
    if thresholds is None:
        best_f1s = [0.0] * args.num_classes
        thresholds = [0] * args.num_classes
        for threshold in args.threshold:
            preds_[preds > threshold] = 1
            preds_[preds <= threshold] = 0
            _, _, f1s, _ = precision_recall_fscore_support(out_label_ids, preds_)
            for class_id in range(args.num_classes):
                if f1s[class_id] > best_f1s[class_id]:
                    # print(f"class {class_id}, old f1: {best_f1s[class_id]}, new f1: {f1s[class_id]}")
                    thresholds[class_id] = threshold
                    best_f1s[class_id]= f1s[class_id]

    for class_id in range(args.num_classes):
        preds_[preds[:, class_id] > thresholds[class_id], class_id] = 1
        preds_[preds[:, class_id] <= thresholds[class_id], class_id] = 0
        
    result = compute_metrics(out_label_ids, preds_, mode = mode)

    # best_f1 = result['weighted_f1_' + mode]
    results.update(result)
    results['threshold'] = thresholds

    detailed_results = {}

    for emotion in range(args.num_labels):
        detailed_results[str(emotion) + "_tp"] = np.logical_and(preds_[:, emotion] == 1, out_label_ids[:, emotion] == 1).astype(np.float32).sum(axis = 0)
        detailed_results[str(emotion) + "_tn"] = np.logical_and(preds_[:, emotion] == 0, out_label_ids[:, emotion] == 0).astype(np.float32).sum(axis = 0)
        detailed_results[str(emotion) + "_fp"] = np.logical_and(preds_[:, emotion] == 1, out_label_ids[:, emotion] == 0).astype(np.float32).sum(axis = 0)
        detailed_results[str(emotion) + "_fn"] = np.logical_and(preds_[:, emotion] == 0, out_label_ids[:, emotion] == 1).astype(np.float32).sum(axis = 0)

    results[f'roc_auc_{mode}'] = roc_auc_score(out_label_ids, preds, average = args.auc_av)

    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir, "{}-{}.txt".format(mode, epoch) if epoch else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(results.keys()):
            logger.info("  {} = {}".format(key, str(results[key])))
            f_w.write("  {} = {}\n".format(key, str(results[key])))

    output_eval_file = os.path.join(output_dir, "{}-{}_detailed.txt".format(mode, epoch) if epoch else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(detailed_results.keys()):
            logger.info("  {} = {}".format(key, str(detailed_results[key])))
            f_w.write("  {} = {}\n".format(key, str(detailed_results[key])))

    return results


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
        elif args.model_type == 'minerva_ffnn5':
            model = minerva_ffnn5(args, load_dir = args.output_dir)
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

    ex_classes = model.ex_classes
    ex_reps = model.ex_class_reps
    model.to(args.device)
    outputs = model(model.ex_features)
    logits = outputs['logits'].to('cpu')
    logits = 1 / (1 + torch.exp(-logits))  # Sigmoid

    print(ex_reps)
    print(ex_classes)
    ex_reps = 1 / (1 + torch.exp(-ex_reps))  # Sigmoid
    thresholds = [0.15, 0.375, 0.35, 0.35]
    # preds = torch.zeros_like(ex_classes, dtype = torch.bool)
    # for i in range(args.num_classes):
    #     preds[:, i] = ex_reps[:, i] > thresholds[i]

    # print(f"class corrects: {(preds == ex_classes).to(torch.float).sum(dim = 0)}")

    ex_reps_match_classes = ex_reps == ex_classes
    auc = roc_auc_score(ex_classes.detach().cpu().numpy(), ex_reps.detach().cpu().numpy(), average = None)
    auc_preds = roc_auc_score(ex_classes.detach().cpu().numpy(), logits.detach().cpu().numpy(), average = None)
    auc_preds_av = roc_auc_score(ex_classes.detach().cpu().numpy(), logits.detach().cpu().numpy(), average = args.auc_av)
    print(f"auc: {auc}")
    print(f"auc_preds: {auc_preds}")
    print(f"auc_preds_av: {auc_preds_av}")

    quit()


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
