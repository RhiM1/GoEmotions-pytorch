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

# from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BertConfig
from transformers import BertTokenizer, AutoTokenizer

from model import BertForMultiLabelClassification, BertMinervaForMultiLabelClassification, MinervaConfig, BertMinervaMSEForMultiLabelClassification, STMinervaMSEForMultiLabelClassification
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


def train(args,
          model,
          tokenizer,
          train_dataset,
          dev_dataset=None,
          test_dataset=None):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    
    if args.freeze_transformer:
        print("Freezing transformer model parameters...")
        for param in model.sentence_transformer.parameters():
            param.requires_grad = False

        optimizer_grouped_parameters = [
            # {'params': [p for n, p in model.bert.named_parameters() if not any(nd in n for nd in no_decay)],
            #  'weight_decay': args.weight_decay,
            #  'lr': args.learning_rate},
            # {'params': [p for n, p in model.bert.named_parameters() if any(nd in n for nd in no_decay)], 
            #  'weight_decay': 0.0,
            #  'lr': args.learning_rate},
            {'params': model.minerva.g.parameters(), 
            'weight_decay': args.weight_decay,
            'lr': args.learning_rate}
        ]

    else:
        print("Training transformer model parameters...")
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.sentence_transformer.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay,
             'lr': args.learning_rate},
            {'params': [p for n, p in model.sentence_transformer.named_parameters() if any(nd in n for nd in no_decay)], 
             'weight_decay': 0.0,
             'lr': args.learning_rate},
            {'params': model.minerva.g.parameters(), 
            'weight_decay': args.weight_decay,
            'lr': args.learning_rate}
    ]

    if args.minerva_train_class_reps:
        optimizer_grouped_parameters.append(
            {'params': model.minerva.class_reps, 
            'weight_decay': 0.0,
            'lr': args.minerva_lr}
        )
    if args.minerva_train_ex_class:
        optimizer_grouped_parameters.append(
            {'params': model.minerva.ex_class_add, 
            'weight_decay': 0.0,
            'lr': args.minerva_lr}
        )

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=args.adam_epsilon)
    # optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(t_total * args.warmup_proportion),
        num_training_steps=t_total
    )

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
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    tr_loss = 0.0

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "labels": batch[3]
            }
            outputs = model(**inputs)

            loss = outputs[0]

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

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    
                    if not cli_args.skip_wandb:
                        wandb.log({
                            'train_loss': tr_loss / global_step,
                            'global_step': global_step
                        })

                    results = evaluate(args, model, dev_dataset, "dev", global_step)
                    if args.evaluate_test_during_training:
                        evaluate(args, model, test_dataset, "test", global_step, threshold = results["threshold"])
                    

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # save class reps (if being trained)
                    if args.minerva_train_class_reps:
                        with open(args.output_dir + "/class_reps.txt", 'a') as f:
                            f.write(f"global step: {global_step}\n")
                            f.write(f"{model.minerva.class_reps}\n")

                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to {}".format(output_dir))

                    if args.save_optimizer:
                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to {}".format(output_dir))

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset, mode, global_step=None, threshold = None):
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
        str(mode) + "_loss": eval_loss
    }
    
    preds_ = np.copy(preds)
    print(f"\npreds_\n{preds_}\n")
    print(f"\nclass_reps:\n{model.minerva.class_reps}\n")

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

                for emotion in range(model.num_labels):
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
        # if result['weighted_f1_' + mode] > best_f1:
        #     best_f1 = result['weighted_f1_' + mode]
        results.update(result)
            # results['threshold'] = threshold
        detailed_results = {}

        for emotion in range(model.num_labels):
            detailed_results[str(emotion) + "_tp"] = np.logical_and(preds_[:, emotion] == 1, out_label_ids[:, emotion] == 1).astype(np.float32).sum(axis = 0)
            detailed_results[str(emotion) + "_tn"] = np.logical_and(preds_[:, emotion] == 0, out_label_ids[:, emotion] == 0).astype(np.float32).sum(axis = 0)
            detailed_results[str(emotion) + "_fp"] = np.logical_and(preds_[:, emotion] == 1, out_label_ids[:, emotion] == 0).astype(np.float32).sum(axis = 0)
            detailed_results[str(emotion) + "_fn"] = np.logical_and(preds_[:, emotion] == 0, out_label_ids[:, emotion] == 1).astype(np.float32).sum(axis = 0)
        
        print(detailed_results)

    if not cli_args.skip_wandb:
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


def main(cli_args):
    # Read from config file and make args
    config_filename = "{}.json".format(cli_args.taxonomy)
    with open(os.path.join("config", config_filename)) as f:
        args = AttrDict(json.load(f))
    logger.info("Training/evaluation parameters {}".format(args))

    try:
        x = args.minerva_lr
    except:
        print("Using global learning rate for Minerva")
        args.minerva_lr = args.learning_rate

    
    try:
        x = args.minerva_m
    except:
        args.minerva_m = 0.5

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
        train_ex_feat = args.minerva_train_ex_feats,
        m = args.minerva_m,
        class_init = torch.tensor(args.class_init, dtype = torch.float),
        num_classes = len(label_list)
    )

    tokenizer = AutoTokenizer.from_pretrained(
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
    model = STMinervaMSEForMultiLabelClassification(
        # args.model_name_or_path,
        config=config,
        minerva_config = minerva_config,
        exemplars = exemplars
    )

    model.to(args.device)

    if dev_dataset is None:
        args.evaluate_test_during_training = True  # If there is no dev dataset, only use test dataset

    if not cli_args.skip_wandb:
        modelName = args.output_dir.split("-")[2] + \
            "_" + args.model_type + \
            "_ex" + str(args.num_ex) + \
            "_wd" + str(args.weight_decay) + \
            "_lr" + str(args.learning_rate) + \
            "_mlr" + str(args.minerva_lr) + \
            "_m" + str(args.minerva_m) + \
            "_p" + str(args.minerva_p_factor) + \
            "_ug" + str(int(args.minerva_use_g)) + \
            "_dr" + str(int(args.minerva_dim_reduce)) + \
            "_cd" + str(args.minerva_class_dim) + \
            "_tcr" + str(int(args.minerva_train_class_reps)) + \
            "_tec" + str(int(args.minerva_train_ex_class))
        run = wandb.init(project=args.wandb_project, reinit = True, name = modelName)

        print(f'\nLogging with Wandb id: {wandb.run.id}\n')

        wandb_config={
            "weight decay": args.weight_decay,
            "p-factor": args.minerva_p_factor,
            "num ex": args.num_ex,
            "dataset": "GoEmotions",
            "epochs": args.num_train_epochs,
            "weight_decay": args.weight_decay,
            "learning rate": args.learning_rate,
            "minerva learning rate": args.minerva_lr,
            "use_g": args.minerva_use_g,
            "minerva dropout": args.minerva_dropout,
            "train_class_reps": args.minerva_train_class_reps,
            "train_ex_class": args.minerva_train_ex_class,
            "train_ex_feats": args.minerva_train_ex_feats
        }

    if args.do_train:
        global_step, tr_loss = train(args, model, tokenizer, train_dataset, dev_dataset, test_dataset)
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
            model = STMinervaMSEForMultiLabelClassification.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, test_dataset, mode="test", global_step=global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f_w:
            for key in sorted(results.keys()):
                f_w.write("{} = {}\n".format(key, str(results[key])))

    
    if not cli_args.skip_wandb:
        run.finish()


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--taxonomy", type=str, required=True, help="Taxonomy (original, ekman, group)")
    cli_parser.add_argument("--skip_wandb", action='store_true', help="Don't use WandB logging")

    cli_args = cli_parser.parse_args()

    main(cli_args)
