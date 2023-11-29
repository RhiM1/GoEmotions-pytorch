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

from transformers import BertConfig, BertTokenizer, AdamW, \
    get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from model import ffnn_wrapper, minerva, \
    BertForMultiLabelClassification, BertMinervaForMultiLabelClassification

from utils import init_logger, set_seed, compute_metrics
from data_loader import load_and_cache_examples, GoEmotionsProcessor
from w2v_dataset import get_word2vec_dataset
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

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_epochs * len(train_dataloader) // args.gradient_accumulation_steps,
        num_training_steps=t_total
    )
    # scheduler = get_constant_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=args.warmup_epochs * len(train_dataloader) // args.gradient_accumulation_steps
    # )

    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    # logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    tr_loss = 0
    best_weighted_f1_dev = 0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    for _ in train_iterator:
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
            # loss = outputs[0]

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
                scheduler.step()
                model.zero_grad()
                global_step += 1

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    
                    if not args.skip_wandb:
                        wandb.log({
                            'train_loss': tr_loss / global_step,
                            'global_step': global_step
                        })

                    results = evaluate(args, model, dev_dataset, "dev", ex_dataset = train_dataset, global_step = global_step)
                    if args.evaluate_test_during_training:
                        evaluate(args, model, test_dataset, "test", ex_dataset = train_dataset, global_step = global_step, threshold = results["threshold"])

                # if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    if results['weighted_f1_dev'] > best_weighted_f1_dev:

                        output_dir = os.path.join(args.output_dir, "checkpoint")
                        # output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
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
                            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            logger.info("Saving optimizer and scheduler states to {}".format(output_dir))


                        # # test code!
                        # model_to_save.load_pretrained(output_dir)
                        # evaluate(args, model, test_dataset, "test", ex_dataset = train_dataset, global_step = global_step, threshold = results["threshold"])

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset, mode, ex_dataset = None, global_step=None, threshold = None):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)
    if args.model_type == 'minerva' and not args.fix_ex:
        total_ex = len(ex_dataset)
        ex_so_far = 0
        ex_dataloader = iter(DataLoader(ex_dataset, sampler=RandomSampler(ex_dataset), batch_size=args.num_ex))

    # Eval!
    if global_step != None:
        logger.info("***** Running evaluation on {} dataset ({} step) *****".format(mode, global_step))
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
            # tmp_eval_loss, logits = outputs[:2]

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
            # print(f"threshold: {threshold}, weighted_f1: {result['weighted_f1']:>0.2f}")
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
        # print(f"threshold: {threshold}, weighted_f1: {result['weighted_f1']:>0.2f}")
        results.update(result)

        detailed_results = {}

        for emotion in range(args.num_labels):
            detailed_results[str(emotion) + "_tp"] = np.logical_and(preds_[:, emotion] == 1, out_label_ids[:, emotion] == 1).astype(np.float32).sum(axis = 0)
            detailed_results[str(emotion) + "_tn"] = np.logical_and(preds_[:, emotion] == 0, out_label_ids[:, emotion] == 0).astype(np.float32).sum(axis = 0)
            detailed_results[str(emotion) + "_fp"] = np.logical_and(preds_[:, emotion] == 1, out_label_ids[:, emotion] == 0).astype(np.float32).sum(axis = 0)
            detailed_results[str(emotion) + "_fn"] = np.logical_and(preds_[:, emotion] == 0, out_label_ids[:, emotion] == 1).astype(np.float32).sum(axis = 0)

        print(f"threshold: {threshold} \n{detailed_results}")
        
    if not args.skip_wandb:
        wandb.log(results)

    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir, "{}-{}.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(results.keys()):
            logger.info("  {} = {}".format(key, str(results[key])))
            f_w.write("  {} = {}\n".format(key, str(results[key])))

    output_eval_file = os.path.join(output_dir, "{}-{}_detailed.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(detailed_results.keys()):
            logger.info("  {} = {}".format(key, str(detailed_results[key])))
            f_w.write("  {} = {}\n".format(key, str(detailed_results[key])))

    return results


def main(args):

    # print(args)

    logger.info("Training/evaluation parameters {}".format(args))
    init_logger()
    set_seed(args)

    if args.feats_type == 'bert':

        processor = GoEmotionsProcessor(args)
        label_list = processor.get_labels()
        args.num_labels = len(label_list)

        config = BertConfig.from_pretrained(
            args.model_name_or_path,
            num_labels = len(label_list),
            finetuning_task = args.task,
            id2label = {str(i): label for i, label in enumerate(label_list)},
            label2id = {label: i for i, label in enumerate(label_list)},
            class_dim = args.class_dim
        )
        
        args.input_dim = config.hidden_size

        tokenizer = BertTokenizer.from_pretrained(
            args.tokenizer_name_or_path,
        )

        # Load dataset
        train_dataset = load_and_cache_examples(args, tokenizer, mode="train") if args.train_file else None
        dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev") if args.dev_file else None
        test_dataset = load_and_cache_examples(args, tokenizer, mode="test") if args.test_file else None

        if args.model_type == 'ffnn':
            model = BertForMultiLabelClassification.from_pretrained(
                args.model_name_or_path,
                config = config,
                args = args
            )
        elif args.model_type == 'minerva':

            # Get the exemplars, if necessary
            if args.fix_ex:
                ex_IDX = torch.randperm(len(train_dataset))[0:args.num_ex]
                exemplars = train_dataset[ex_IDX]
                exemplars = [exemplar.to(args.device) for exemplar in exemplars]
            else:
                exemplars = None
                ex_IDX = None

            model = BertMinervaForMultiLabelClassification(
                config = config,
                args = args,
                exemplars = exemplars,
                ex_IDX = ex_IDX
            )
    elif args.feats_type == 'lsa' or args.feats_type == 'word2vec':
        tokenizer = None
        if args.feats_type == 'lsa':
            feats_model = get_lsa_dict('data/LSA/TASA.rda')
        elif args.feats_type == "word2vec":
            feats_model = api.load(args.model_name_or_path)

        # Load dataset
        train_dataset = get_word2vec_dataset(args.data_dir, mode = 'train', model = feats_model) if args.train_file else None
        dev_dataset = get_word2vec_dataset(args.data_dir, mode = 'dev', model = feats_model) if args.dev_file else None
        test_dataset = get_word2vec_dataset(args.data_dir, mode = 'test', model = feats_model) if args.test_file else None

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
                f"fdo{args.dropout}",
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

    if args.do_train:
        global_step, tr_loss = train(args, model, train_dataset, tokenizer, dev_dataset, test_dataset)
        logger.info(" global_step = {}, average loss = {}".format(global_step, tr_loss))

    results = {}
    if args.do_eval:
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin", recursive=True))
        )
        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1]
            model = BertMinervaForMultiLabelClassification.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, test_dataset, mode="test", global_step=global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f_w:
            for key in sorted(results.keys()):
                f_w.write("{} = {}\n".format(key, str(results[key])))

    if not args.skip_wandb:
        run.finish()


if __name__ == '__main__':

    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument("--config", type=str, required=True, help="name of the configuration file")
    arg_parser.add_argument("--skip_wandb", default=False, action='store_true', help="Don't use WandB logging")

    cli_args = arg_parser.parse_args()

    # Read from config file and make args
    config_filename = f"{cli_args.config}.json"
    with open(os.path.join("config", config_filename)) as f:
        args = AttrDict(json.load(f))

    args.data_dir = DATAROOT
    args.train_file = "train.tsv"
    args.dev_file = "dev.tsv"
    args.test_file = "test.tsv"
    args.label_file = "labels.txt"

    args.skip_wandb = cli_args.skip_wandb
    args.output_dir = f"{args.ckpt_dir}/{args.feats_type}_{args.model_type}_{args.exp_id}_{args.run_id}"
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    main(args)
