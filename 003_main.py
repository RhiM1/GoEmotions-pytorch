import argparse
import json
import logging
import os
import glob
import wandb

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from attrdict import AttrDict
import gensim.downloader as api

# from transformers import BertConfig, BertTokenizer, AdamW, \
#     get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
from sentence_transformers import SentenceTransformer, util

from model import ffnn_wrapper, minerva, \
    BertForMultiLabelClassification, BertMinervaForMultiLabelClassification

from utils import init_logger, set_seed, compute_metrics
from data_loader import load_and_cache_examples, GoEmotionsProcessor
from w2v_dataset import get_goem_dataset
from LSA_process import get_lsa_dict

from constants import DATAROOT

logger = logging.getLogger(__name__)

def train(
        args,
        model,
        train_dataset,
        tokenizer= None,
        dev_dataset = None,
        test_dataset = None
    ):

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    if args.model_type == 'minerva' and not args.fix_ex:
        total_ex = len(train_dataset)
        ex_so_far = 0
        ex_dataloader = iter(DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.num_ex))

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    optParams = []
    for name, param in model.named_parameters():
        if name == "class_reps":
            optParams.append(
                {'params': param, 'weight_decay': args.wd_ex}
            )
        elif name == "add_ex_reps" or name == "add_ex_feats":
            optParams.append(
                {'params': param, 'weight_decay': args.weight_decay}
            )
        else:
            optParams.append(
                {'params': param, 'weight_decay': args.weight_decay}
            )

    
    # optimizer = torch.optim.Adam(optParams, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = torch.optim.AdamW(optParams, lr=args.learning_rate, eps=args.adam_epsilon)

    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load optimizer states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Save steps = %d", args.save_steps)

    # global_step = 0
    tr_loss = 0
    best_weighted_f1_dev = 0
    epoch = 1
    best_epoch = 0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    for _ in train_iterator:
        tr_loss = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            if args.feats_type == 'bert':
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    "labels": batch[3]
                }
                if args.model_type == 'minerva' and not args.fix_ex:
                    ex_so_far += args.num_ex
                    if ex_so_far > total_ex:
                        ex_dataloader = iter(DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.num_ex))
                        ex_so_far = args.num_ex
                    exemplars = next(ex_dataloader)
                    inputs['exemplars'] = [exemplar.to(args.device) for exemplar in exemplars]

            else:
                inputs = {
                    "features": batch[0],
                    "labels": batch[1]
                }
                if args.model_type == 'minerva' and not args.fix_ex:
                    ex_so_far += args.num_ex
                    if ex_so_far > total_ex:
                        ex_dataloader = iter(DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=args.num_ex))
                        ex_so_far = args.num_ex

                    ex_features, ex_classes = next(ex_dataloader)
                    inputs['ex_features'] = ex_features.to(args.device)
                    inputs['ex_classes'] = ex_classes.to(args.device)

            loss, _ = model(**inputs)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    len(train_dataloader) <= args.gradient_accumulation_steps
                    and (step + 1) == len(train_dataloader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                model.zero_grad()
                # global_step += 1

        results = evaluate(args, model, dev_dataset, "dev", ex_dataset = train_dataset, epoch = epoch)
        if args.evaluate_test_during_training:
            test_results = evaluate(args, model, test_dataset, "test", ex_dataset = train_dataset, epoch = epoch, threshold = results["threshold"])
            results.update(test_results)
        results['train_loss'] = tr_loss / len(train_dataloader)
        results['epoch'] = epoch
            
        if not args.skip_wandb:
            wandb.log(results)

        if results['weighted_f1_dev'] > best_weighted_f1_dev:

            best_weighted_f1_dev = results['weighted_f1_dev']
            best_epoch = epoch

            output_dir = os.path.join(args.output_dir, "checkpoint")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )
            model_to_save.save_pretrained(output_dir)
            if tokenizer is not None:
                tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to {}".format(output_dir))

            if args.save_optimizer:
                torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                logger.info("Saving optimizer and scheduler states to {}".format(output_dir))

        epoch += 1

    return best_epoch


def evaluate(args, model, eval_dataset, mode, ex_dataset = None, epoch=None, threshold = None):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    if args.model_type == 'minerva' and not args.fix_ex:
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
                if args.model_type == 'minerva' and not args.fix_ex:
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
                if args.model_type == 'minerva' and not args.fix_ex:
                    ex_so_far += args.num_ex
                    if ex_so_far > total_ex:
                        ex_dataloader = iter(DataLoader(ex_dataset, sampler=RandomSampler(ex_dataset), batch_size=args.num_ex))
                        ex_so_far = args.num_ex
                    ex_features, ex_classes = next(ex_dataloader)
                    inputs['ex_features'] = ex_features.to(args.device)
                    inputs['ex_classes'] = ex_classes.to(args.device)

            tmp_eval_loss, logits = model(**inputs)

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = 1 / (1 + np.exp(-logits.detach().cpu().numpy()))  # Sigmoid
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, 1 / (1 + np.exp(-logits.detach().cpu().numpy())), axis=0)  # Sigmoid
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    results = {
        str(mode) + "_loss": eval_loss
    }
    
    preds_ = np.copy(preds)
    if threshold is None:
        best_f1 = 0.0
        for threshold in args.threshold:

            preds_[preds > threshold] = 1
            preds_[preds <= threshold] = 0
            result = compute_metrics(out_label_ids, preds_, mode = mode)
            if result['weighted_f1_' + mode] > best_f1:
                best_f1 = result['weighted_f1_' + mode]
                results.update(result)
                results['threshold'] = threshold

                detailed_results = {}

                for emotion in range(args.num_labels):
                    detailed_results[str(emotion) + "_tp"] = np.logical_and(preds_[:, emotion] == 1, out_label_ids[:, emotion] == 1).astype(np.float32).sum(axis = 0)
                    detailed_results[str(emotion) + "_tn"] = np.logical_and(preds_[:, emotion] == 0, out_label_ids[:, emotion] == 0).astype(np.float32).sum(axis = 0)
                    detailed_results[str(emotion) + "_fp"] = np.logical_and(preds_[:, emotion] == 1, out_label_ids[:, emotion] == 0).astype(np.float32).sum(axis = 0)
                    detailed_results[str(emotion) + "_fn"] = np.logical_and(preds_[:, emotion] == 0, out_label_ids[:, emotion] == 1).astype(np.float32).sum(axis = 0)

                print(f"threshold: {threshold} \n{detailed_results}")

    else:
        preds_[preds > threshold] = 1
        preds_[preds <= threshold] = 0
        result = compute_metrics(out_label_ids, preds_, mode = mode)
        results.update(result)

        detailed_results = {}

        for emotion in range(args.num_labels):
            detailed_results[str(emotion) + "_tp"] = np.logical_and(preds_[:, emotion] == 1, out_label_ids[:, emotion] == 1).astype(np.float32).sum(axis = 0)
            detailed_results[str(emotion) + "_tn"] = np.logical_and(preds_[:, emotion] == 0, out_label_ids[:, emotion] == 0).astype(np.float32).sum(axis = 0)
            detailed_results[str(emotion) + "_fp"] = np.logical_and(preds_[:, emotion] == 1, out_label_ids[:, emotion] == 0).astype(np.float32).sum(axis = 0)
            detailed_results[str(emotion) + "_fn"] = np.logical_and(preds_[:, emotion] == 0, out_label_ids[:, emotion] == 1).astype(np.float32).sum(axis = 0)

        print(f"threshold: {threshold} \n{detailed_results}")
        
    # if not args.skip_wandb:
    #     wandb.log(results)

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


def main(args):

    logger.info("Training/evaluation parameters {}".format(args))
    init_logger()
    set_seed(args)

    tokenizer = None
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

    example_feats, example_classes = train_dataset[0]
    print(f"features size: {example_feats.size(0)}, num_classes: {example_classes.size(0)}")
    args.input_dim = example_feats.size(0)
    args.num_labels = example_classes.size(0)

    if args.model_type == 'ffnn':
        model = ffnn_wrapper(args)
    elif args.model_type == 'minerva':
        if args.fix_ex:
            ex_IDX = torch.randperm(len(train_dataset))[0:args.num_ex]
            exemplars = train_dataset[ex_IDX]
            ex_features = exemplars[0]
            ex_classes = exemplars[1]
        else:
            ex_IDX = None
            ex_features = None
            ex_classes = None
        model = minerva(
            args,
            ex_classes = ex_classes,
            ex_features = ex_features,
            ex_IDX = ex_IDX
        )

    model.to(args.device)

    if dev_dataset is None:
        args.evaluate_test_during_training = True  # If there is no dev dataset, only use test dataset
        
    if not args.skip_wandb:
        modelName = args.output_dir.split("/")[-1]
        run = wandb.init(
            project=args.wandb_project, 
            reinit = True, 
            name = modelName,
            tags = [
                f"e{args.exp_id}",
                f"r{args.run_id}",
                f"{args.feats_type}",
                f"{args.model_type}",
                f"bs{args.train_batch_size * args.gradient_accumulation_steps}", 
                f"lr{args.learning_rate}", 
                f"wd{args.weight_decay}",
                f"fdo{args.do_feat}",
                f"cdo{args.do_class}",
                f"fd{args.feat_dim}",
                f"cd{args.class_dim}",
                f"s{args.seed}"
            ]
        )
        if args.model_type == 'minerva':
            run.tags = run.tags + (
                f"g{int(args.use_g)}",
                f"ex{args.num_ex}",
                f"p{args.p_factor}",
                f"tcr{int(args.train_class_reps)}",
                f"tec{int(args.train_ex_class)}",
                f"tef{int(args.train_ex_feats)}",
                f"wde{args.wd_ex}",
                f"fe{int(args.fix_ex)}"
            )
    
    init_dev_results = evaluate(args, model, dev_dataset, mode="dev", epoch=0)
    wandb.log(init_dev_results)

    if args.do_train:
        best_epoch = train(args, model, train_dataset, tokenizer, dev_dataset, test_dataset)
        # logger.info(" global_step = {}, average loss = {}".format(global_step, tr_loss))

    results = {}
    if args.do_eval:
        # checkpoints = list(
        #     os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin", recursive=True))
        # )
        # if not args.eval_all_checkpoints:
        #     checkpoints = checkpoints[-1:]
        # else:
        #     logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
        #     logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        # logger.info("Evaluate the following checkpoints: %s", checkpoints)
        # for checkpoint in checkpoints:


        # global_step = checkpoint.split("-")[-1]
        # model = BertMinervaForMultiLabelClassification.from_pretrained(checkpoint)
        model.load_pretrained(args.output_dir + "/checkpoint")
        model.to(args.device)
        result = evaluate(args, model, test_dataset, mode="test", epoch=best_epoch)
        result = dict((k + "_{}".format(best_epoch), v) for k, v in result.items())
        results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f_w:
            for key in sorted(results.keys()):
                f_w.write("{} = {}\n".format(key, str(results[key])))

    if not args.skip_wandb:
        run.finish()


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()

    # General args
    arg_parser.add_argument(
        "--exp_id", help = "experiment id", default = "001"
    )
    arg_parser.add_argument(
        "--run_id", help = "ID of current run", default = "test"
    )
    # arg_parser.add_argument("--config", type=str, required=True, help="name of the configuration file")
    arg_parser.add_argument(
        "--skip_wandb", help="Don't use WandB logging", default=False, action='store_true'
    )
    arg_parser.add_argument(
        "--wandb_project", help = "WandB project name", default = "CSL_goemotions"
    )
    arg_parser.add_argument(
        "--evaluate_test_during_training", help="", default=True, action='store_true'
    )
    arg_parser.add_argument(
        "--eval_all_checkpoints", help="", default=True, action='store_true'
    )
    arg_parser.add_argument(
        "--save_optimizer", help="", default=False, action='store_true'
    )
    arg_parser.add_argument(
        "--do_lower_case", help="", default=False, action='store_true'
    )
    arg_parser.add_argument(
        "--do_train", help="", default=True, action='store_true'
    )
    arg_parser.add_argument(
        "--do_eval", help="", default=True, action='store_true'
    )
    arg_parser.add_argument(
        "--seed", help = "only used for BERT", default = 42
    )
    arg_parser.add_argument(
        "--save_steps", help = "only used for BERT", default = 1000, type = int
    )
    arg_parser.add_argument(
        "--ckpt_dir", help = "only used for BERT", default = "ckpt/csl_paper"
    )

    # Feats args
    arg_parser.add_argument(
        "--feats_type", help = "feat extraction model: lsa, word2vec, sen_trans", default = "sen_trans"
    )
    arg_parser.add_argument(
        "--model_name_or_path", help = "", default = None
    )

    # Model args
    arg_parser.add_argument(
        "--model_type", help = "ffnn or minerva", default = None
    )
    arg_parser.add_argument(
        "--feat_dim", help = "model feature embeddings dimension, where applicable", default = None, type = int
    )
    arg_parser.add_argument(
        "--class_dim", help = "model class embeddings dimension, where applicable", default = None, type = int
    )
    arg_parser.add_argument(
        "--use_g", help="", default=False, action='store_true'
    )
    arg_parser.add_argument(
        "--p_factor", help = "minerva p factor", default = 1, type = int
    )
    arg_parser.add_argument(
        "--train_class_reps", help="", default=False, action='store_true'
    )
    arg_parser.add_argument(
        "--train_ex_class", help="", default=False, action='store_true'
    )
    arg_parser.add_argument(
        "--train_ex_feats", help="", default=False, action='store_true'
    )
    arg_parser.add_argument(
        "--num_ex", help = "number of exemplars to use for minerva", default = None, type = int
    )
    arg_parser.add_argument(
        "--fix_ex", help="feix the exemplar set, rather than changing per minibatch", default=False, action='store_true'
    )

    # Hyperparameters
    arg_parser.add_argument(
        "--max_seq_len", help = "only used for BERT", default = 50, type = int
    )
    arg_parser.add_argument(
        "--num_train_epochs", help = "number of training epochs", default = 10, type = int
    )
    arg_parser.add_argument(
        "--warmup_epochs", help = "number of training epochs", default = 1, type = int
    )
    arg_parser.add_argument(
        "--weight_decay", help = "number of training epochs", default = 0, type = float
    )
    arg_parser.add_argument(
        "--wd_ex", help = "weight decay for the exemplar represenations, minerva only", default = 0, type = float
    )
    arg_parser.add_argument(
        "--gradient_accumulation_steps", help = "", default = 1, type = int
    )
    arg_parser.add_argument(
        "--train_batch_size", help = "", default = 16, type = int
    )
    arg_parser.add_argument(
        "--eval_batch_size", help = "", default = 32, type = int
    )
    arg_parser.add_argument(
        "--adam_epsilon", help = "", default = 1e-8, type = float
    )
    arg_parser.add_argument(
        "--max_steps", help = "maximum number of training steps, use -1 to use epochs instead", default = -1, type = int
    )
    arg_parser.add_argument(
        "--max_grad_norm", help = "", default = 1.0, type = float
    )
    arg_parser.add_argument(
        "--learning_rate", help = "", default = 1e-5, type = float
    )
    arg_parser.add_argument(
        "--do_feat", help = "", default = 0, type = float
    )
    arg_parser.add_argument(
        "--do_class", help = "", default = 0, type = float
    )
    arg_parser.add_argument(
        "--no_cuda", help="", default=False, action='store_true'
    )

    args = arg_parser.parse_args()

    args.data_dir = DATAROOT
    args.train_file = "train.tsv"
    args.dev_file = "dev.tsv"
    args.test_file = "test.tsv"
    args.label_file = "labels.txt"
    args.threshold = [i/10 for i in range(-5, 6, 1)]
    if args.feats_type == 'sen_trans':
        args.model_name_or_path = "all-MiniLM-L6-v2"
    elif args.feats_type == 'word2vec':
        args.model_name_or_path = "word2vec-google-news-300"

    # maybe delete...
    args.tokenizer_name_or_path = "monologg/bert-base-cased-goemotions-group"
    args.task = "goemotions"

    args.output_dir = f"{args.ckpt_dir}/{args.feats_type}_{args.model_type}_{args.exp_id}_{args.run_id}"
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    main(args)
