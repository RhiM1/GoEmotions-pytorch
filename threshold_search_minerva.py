import argparse
import json
import logging
import os
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange
from attrdict import AttrDict

from transformers import (
    BertConfig,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)

from model import BertForMultiLabelClassification, BertMinervaForMultiLabelClassification, MinervaConfig
from utils import (
    init_logger,
    set_seed,
    compute_metrics
)
from data_loader import (
    load_and_cache_examples,
    GoEmotionsProcessor
)

logger = logging.getLogger(__name__)


def evaluate(args, model, eval_dataset, mode, global_step=None):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

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
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3]
            }
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

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
        "eval_loss": eval_loss
    }

    preds_ = np.copy(preds)

    for threshold in args.threshold:
        # threshold = args.threshold

        # print(f"threshold: {threshold}")
        # print(f"preds_:\n{preds_}")
        # print(f"preds:\n{preds}")

        preds_[preds > threshold] = 1
        preds_[preds <= threshold] = 0
        result = compute_metrics(out_label_ids, preds_, threshold)
        results.update(result)
        print(results)


        tp = np.logical_and(preds_ == 1, out_label_ids == 1).astype(np.float32).sum(axis = 0)
        # print(f"tp.shape: {tp.shape}")
        tn = np.logical_and(preds_ == 0, out_label_ids == 0).astype(np.float32).sum(axis = 0)
        fp = np.logical_and(preds_ == 1, out_label_ids == 0).astype(np.float32).sum(axis = 0)
        fn = np.logical_and(preds_ == 0, out_label_ids == 1).astype(np.float32).sum(axis = 0)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = ((precision**(-1) + recall**(-1))/2)**(-1)

        print(f"Emotion \t Precision \t Recall \t f1 \t tp \t tn \t fp \t fn")
        for emotion in range(len(tp)):
            print(f"{emotion} \t   {precision[emotion]:>0.2f} \t   {recall[emotion]:>0.2f} \t   {f1[emotion]:>0.2f} \t {tp[emotion]:>0.0f} \t {tn[emotion]:>0.0f} \t {fp[emotion]:>0.0f} \t {fn[emotion]:>0.0f}")
        


    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir, "{}-{}.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(results.keys()):
            logger.info("  {} = {}".format(key, str(results[key])))
            f_w.write("  {} = {}\n".format(key, str(results[key])))

    return results


def main(cli_args):
    # Read from config file and make args
    config_filename = "{}.json".format(cli_args.taxonomy)
    with open(os.path.join("config", config_filename)) as f:
        args = AttrDict(json.load(f))
    logger.info("Training/evaluation parameters {}".format(args))

    print(args)

    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

    init_logger()
    set_seed(args)

    processor = GoEmotionsProcessor(args)
    label_list = processor.get_labels()

    config = BertConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_list),
        finetuning_task=args.task,
        id2label={str(i): label for i, label in enumerate(label_list)},
        label2id={label: i for i, label in enumerate(label_list)},
        minerva_dim_reduce = args.minerva_dim_reduce if args.minerva_dim_reduce else None
    )

    minerva_config = MinervaConfig(
        input_dim = config.hidden_size,
        dim_reduce = args.minerva_dim_reduce,
        class_dim = args.minerva_class_dim,
        use_g = args.minerva_use_g,
        dropout = args.minerva_dropout,
        p_factor = args.minerva_p_factor,
        train_class_reps = args.minerva_train_class_reps,
        train_ex_class = args.minerva_train_ex_class,
        train_ex_feat = args.minerava_train_ex_feats,
        num_classes = len(label_list)
    )

    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
    )
    # Load dataset
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train") if args.train_file else None

    exIDX = torch.randperm(len(train_dataset))[0:args.num_ex]

    exemplars = train_dataset[exIDX]

    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev") if args.dev_file else None
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test") if args.test_file else None

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    exemplars = [thing.to(args.device) for thing in exemplars]
    model = BertMinervaForMultiLabelClassification.from_pretrained(
        args.model_name_or_path,
        config=config,
        minerva_config = minerva_config,
        exemplars = exemplars
    )

    model.to(args.device)

    if dev_dataset is None:
        args.evaluate_test_during_training = True  # If there is no dev dataset, only use test dataset

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
            model = BertMinervaForMultiLabelClassification.from_pretrained(checkpoint, minerva_config = minerva_config, exemplars = exemplars)
            model.to(args.device)
            result = evaluate(args, model, test_dataset, mode=cli_args.set, global_step=global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, cli_args.set + "_thresh_search.txt")
        with open(output_eval_file, "w") as f_w:
            for key in sorted(results.keys()):
                f_w.write("{} = {}\n".format(key, str(results[key])))


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--taxonomy", type=str, required=True, help="Taxonomy (original, ekman, group)")
    cli_parser.add_argument("--set", type=str, required=True, help="train, dev or test")

    cli_args = cli_parser.parse_args()

    main(cli_args)
