# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""


import argparse
import glob
import logging
import os
import random
import timeit

import numpy as np
import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import transformers
from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.squad import SquadResult
from transformers.trainer_utils import is_main_process


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from squad.squad import squad_convert_examples_to_features

logger = logging.getLogger(__name__)

from modeling_bert_skim import BertForQuestionAnswering as BertForQuestionAnsweringWithSkim
from modeling_albert_skim import AlbertForQuestionAnswering as AlbertForQuestionAnsweringWithSkim
from modeling_blockskim import compute_skim_mask
from squad.transformer_squad_processor import SquadV1Processor, SquadV2Processor

import gzip
import json

from mrqa.mrqa_official_eval import exact_match_score, f1_score, metric_max_over_ground_truths
from mrqa.run_mrqa_util import make_predictions, get_raw_scores, make_eval_dict
from mrqa.util_squad import convert_examples_to_features
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
                              
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

class MRQAExample(object):
    """
    A single training/test example for the MRQA dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = False

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        return s

def read_mrqa_examples(input_file, is_training):
    """Read a MRQA json file into a list of MRQAExample."""
    with gzip.GzipFile(input_file, 'r') as reader:
        # skip header
        content = reader.read().decode('utf-8').strip().split('\n')[1:]
        input_data = [json.loads(line) for line in content]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    num_answers = 0
    for i, entry in enumerate(input_data):
        if i % 1000 == 0:
            logger.info("Processing %d / %d.." % (i, len(input_data)))
        paragraph_text = entry["context"]
        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True
        for c in paragraph_text:
            if is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        for qa in entry["qas"]:
            qas_id = qa["qid"]
            question_text = qa["question"]
            start_position = None
            end_position = None
            orig_answer_text = None
            answers = qa["detected_answers"]
            # import ipdb
            # ipdb.set_trace()
            spans = sorted([span for spans in answers for span in spans['char_spans']])
            # take first span
            char_start, char_end = spans[0][0], spans[0][1]
            orig_answer_text = paragraph_text[char_start:char_end+1]
            start_position, end_position = char_to_word_offset[char_start], char_to_word_offset[char_end]
            num_answers += sum([len(spans['char_spans']) for spans in answers])

            example = MRQAExample(
                qas_id=qas_id,
                question_text=question_text,
                doc_tokens=doc_tokens,
                orig_answer_text=orig_answer_text,
                start_position=start_position,
                end_position=end_position)
            examples.append(example)
    logger.info('Num avg answers: {}'.format(num_answers / len(examples)))
    return examples

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def train(args, train_dataset, model, tokenizer):
    """Train the model"""
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(args.output_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    if args.block_skim:
        tr_qa_loss, logging_qa_loss, tr_skim_loss, logging_skim_loss = 0.0, 0.0, 0.0, 0.0
        balance_weight = torch.tensor([1, args.balance_factor]).to(args.device)
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
                "start_positions": batch[3],
                "end_positions": batch[4],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
                del inputs["token_type_ids"]

            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[5], "p_mask": batch[6]})
                if args.version_2_with_negative:
                    inputs.update({"is_impossible": batch[7]})
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)

            if args.block_skim:
                if args.actual_skim:
                    new_start_logits = torch.ones(batch[0].shape).to(args.device)*-100
                    new_end_logits = torch.ones(batch[0].shape).to(args.device)*-100
                    final_skim_mask = torch.ones_like(outputs.all_skim_mask[0][1],dtype=torch.bool)

                    for layer_idx, skim_mask_tuple in enumerate(outputs.all_skim_mask):
                        new_final_skim_mask = final_skim_mask.clone()
                        new_final_skim_mask[final_skim_mask] = skim_mask_tuple[1].view(-1)
                        final_skim_mask = new_final_skim_mask

                    new_start_logits[final_skim_mask] = outputs[1].view(-1)
                    new_end_logits[final_skim_mask] = outputs[2].view(-1)
                    loss_fct = torch.nn.CrossEntropyLoss()
                    start_loss = loss_fct(new_start_logits, batch[3])
                    end_loss = loss_fct(new_end_logits, batch[4])
                    qa_loss = (start_loss + end_loss) / 2
                    skim_loss = qa_loss
                    loss = qa_loss
                else:
                    qa_loss = outputs[0]
                    answer_mask = batch[-1]
                    all_skim_mask = outputs[-1]
                    # blocked_answer_mask = answer_mask.view((-1, model.config.max_seq_length//model.config.block_size, model.config.block_size))
                    # skim_label = (torch.sum(blocked_answer_mask, dim=-1)>1).to(dtype=torch.long)
                    skim_label = compute_skim_mask(answer_mask, args.max_seq_length//args.block_size, args.block_size)
                    all_skim_loss = [torch.nn.functional.cross_entropy(skim_mask[0].view(-1,2), skim_label.view(-1), weight=balance_weight) for skim_mask in all_skim_mask]
                    skim_loss = sum(all_skim_loss)

                    loss = args.skim_factor * skim_loss + qa_loss
            else:
                loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
                if args.block_skim:
                    qa_loss = qa_loss.mean()
                    skim_loss = skim_loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                if args.block_skim:
                    qa_loss = qa_loss / args.gradient_accumulation_steps
                    skim_loss = skim_loss / args.gradient_accumulation_steps


            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if args.block_skim:
                tr_qa_loss += qa_loss.item()
                tr_skim_loss += skim_loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
                    if args.block_skim:
                        tb_writer.add_scalar("skim_loss", (tr_skim_loss - logging_skim_loss) / args.logging_steps, global_step)
                        tb_writer.add_scalar("qa_loss", (tr_qa_loss - logging_qa_loss) / args.logging_steps, global_step)
                        logging_qa_loss = tr_qa_loss
                        logging_skim_loss = tr_skim_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_results = []
    if args.block_skim:
        num_hidden_layers = model.module.config.num_hidden_layers if isinstance(model, DataParallel) else model.config.num_hidden_layers
        all_layer_skim_mask = [[] for _ in range(num_hidden_layers)]
        all_skim_label = []
    start_time = timeit.default_timer()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            if args.model_type in ["xlm", "roberta", "distilbert", "camembert", "bart", "longformer"]:
                del inputs["token_type_ids"]

            feature_indices = batch[3]

            # XLNet and XLM use more arguments for their predictions
            if args.model_type in ["xlnet", "xlm"]:
                inputs.update({"cls_index": batch[4], "p_mask": batch[5]})
                # for lang_id-sensitive xlm models
                if hasattr(model, "config") and hasattr(model.config, "lang2id"):
                    inputs.update(
                        {"langs": (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)}
                    )
            outputs = model(**inputs)


            if args.block_skim and args.actual_skim:
                new_start_logits = torch.ones(batch[0].shape).to(args.device)*-100
                new_end_logits = torch.ones(batch[0].shape).to(args.device)*-100
                final_skim_mask = torch.ones_like(batch[0],dtype=torch.bool)

                for layer_idx, skim_mask_tuple in enumerate(outputs.all_skim_mask):
                    if skim_mask_tuple[1]==None:
                        continue
                    new_final_skim_mask = final_skim_mask.clone()
                    new_final_skim_mask[final_skim_mask] = skim_mask_tuple[1].view(-1)
                    final_skim_mask = new_final_skim_mask

                new_start_logits[final_skim_mask] = outputs[0].view(-1)
                new_end_logits[final_skim_mask] = outputs[1].view(-1)

                outputs.start_logits = new_start_logits
                outputs.end_logits = new_end_logits


        for i, feature_index in enumerate(feature_indices):
            eval_feature = features[feature_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs.to_tuple() if isinstance(output, torch.Tensor)]

            if args.block_skim:
                skim_label = compute_skim_mask(batch[-1][i], args.max_seq_length//args.block_size, args.block_size)
                for layer_idx, skim_mask in enumerate(all_layer_skim_mask):
                    if outputs.all_skim_mask[layer_idx][0]==None:
                        continue
                    skim_mask.extend(to_list(torch.argmax(outputs.all_skim_mask[layer_idx][0][i],axis=-1)))
                all_skim_label.extend(to_list(skim_label))
                
                # assert len(all_skim_label) == len(all_layer_skim_mask[0])

            # Some models (XLNet, XLM) use 5 arguments for their predictions, while the other "simpler"
            # models only use two.
            if len(output) >= 5:
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(
                    unique_id,
                    start_logits,
                    end_logits,
                    start_top_index=start_top_index,
                    end_top_index=end_top_index,
                    cls_logits=cls_logits,
                )

            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    if args.block_skim and not args.actual_skim:
        from sklearn.metrics import classification_report
        for layer_idx in range(len(all_layer_skim_mask)):
            if args.augment_layers and layer_idx not in args.augment_layers:
                continue
            print(f'evaluating skim predictor of layer {layer_idx}')
            print(classification_report(all_skim_label, all_layer_skim_mask[layer_idx]))

    # Compute predictions
    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

    if args.version_2_with_negative:
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    # # XLNet and XLM use a more complex post-processing procedure
    # if args.model_type in ["xlnet", "xlm"]:
    #     start_n_top = model.config.start_n_top if hasattr(model, "config") else model.module.config.start_n_top
    #     end_n_top = model.config.end_n_top if hasattr(model, "config") else model.module.config.end_n_top

    #     predictions = compute_predictions_log_probs(
    #         examples,
    #         features,
    #         all_results,
    #         args.n_best_size,
    #         args.max_answer_length,
    #         output_prediction_file,
    #         output_nbest_file,
    #         output_null_log_odds_file,
    #         start_n_top,
    #         end_n_top,
    #         args.version_2_with_negative,
    #         tokenizer,
    #         args.verbose_logging,
    #     )
    # else:
    #     predictions = compute_predictions_logits(
    #         examples,
    #         features,
    #         all_results,
    #         args.n_best_size,
    #         args.max_answer_length,
    #         args.do_lower_case,
    #         output_prediction_file,
    #         output_nbest_file,
    #         output_null_log_odds_file,
    #         args.verbose_logging,
    #         args.version_2_with_negative,
    #         args.null_score_diff_threshold,
    #         tokenizer,
    #     )

    with gzip.GzipFile(os.path.join(args.data_dir,args.predict_file), 'r') as reader:
        content = reader.read().decode('utf-8').strip().split('\n')[1:]
        eval_dataset = [json.loads(line) for line in content]

    preds, nbest_preds = \
        make_predictions(examples, features, all_results,
                         args.n_best_size, args.max_answer_length,
                         args.do_lower_case, args.verbose_logging)
    exact_raw, f1_raw = get_raw_scores(eval_dataset, preds)
    result = make_eval_dict(exact_raw, f1_raw)

    # # Compute the F1 and exact scores.
    # results = squad_evaluate(examples, predictions)
    return result

def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    input_file = args.predict_file if evaluate else args.train_file
    cached_features_file = os.path.join(args.data_dir,os.path.dirname(input_file), 'cached_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache :
        logger.info("Loading features from cached file %s", cached_features_file)
        examples = read_mrqa_examples(input_file=os.path.join(args.data_dir,input_file),
                                                is_training=True)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", input_file)
        examples = read_mrqa_examples(input_file=os.path.join(args.data_dir,input_file),
                                                is_training=True)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=True,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                pad_token_segment_id=3 if args.model_type in ['xlnet'] else 0,
                                                cls_token_at_end=True if args.model_type in ['xlnet'] else False,
                                                sequence_a_is_doc=True if args.model_type in ['xlnet'] else False)
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
    all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)
    all_answer_mask = torch.tensor([f.answer_mask for f in features], dtype=torch.float)
    if evaluate:
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_example_index, all_cls_index, all_p_mask, all_answer_mask)
    else:
        all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_start_positions, all_end_positions,
                                all_cls_index, all_p_mask, all_answer_mask)

    if output_examples:
        return dataset, examples, features
    return dataset

# def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
#     if args.local_rank not in [-1, 0] and not evaluate:
#         # Make sure only the first process in distributed training process the dataset, and the others will use the cache
#         torch.distributed.barrier()

#     # Load data features from cache or dataset file
#     input_dir = args.data_dir if args.data_dir else "."
#     cached_features_file = os.path.join(
#         input_dir,
#         "cached_{}_{}_{}".format(
#             "dev" if evaluate else "train",
#             list(filter(None, args.cache_name.split("/"))).pop() if args.cache_name else list(filter(None, args.model_name_or_path.split("/"))).pop(),
#             str(args.max_seq_length),
#         ),
#     )

#     # Init features and dataset from cache if it exists
#     if os.path.exists(cached_features_file) and not args.overwrite_cache:
#         logger.info("Loading features from cached file %s", cached_features_file)
#         features_and_dataset = torch.load(cached_features_file)
#         features, dataset, examples = (
#             features_and_dataset["features"],
#             features_and_dataset["dataset"],
#             features_and_dataset["examples"],
#         )
#     else:
#         logger.info("Creating features from dataset file at %s", input_dir)

#         if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
#             try:
#                 import tensorflow_datasets as tfds
#             except ImportError:
#                 raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

#             if args.version_2_with_negative:
#                 logger.warning("tensorflow_datasets does not handle version 2 of SQuAD.")

#             tfds_examples = tfds.load("squad")
#             examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
#         else:
#             processor = SquadV2Processor() if args.version_2_with_negative else SquadV1Processor()
#             if evaluate:
#                 examples = processor.get_dev_examples(args.data_dir, filename=args.predict_file)
#             else:
#                 examples = processor.get_train_examples(args.data_dir, filename=args.train_file)

#         features, dataset = squad_convert_examples_to_features(
#             examples=examples,
#             tokenizer=tokenizer,
#             max_seq_length=args.max_seq_length,
#             doc_stride=args.doc_stride,
#             max_query_length=args.max_query_length,
#             is_training=not evaluate,
#             return_dataset="pt",
#             threads=args.threads,
#         )

#         if args.local_rank in [-1, 0]:
#             logger.info("Saving features into cached file %s", cached_features_file)
#             torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

#     if args.local_rank == 0 and not evaluate:
#         # Make sure only the first process in distributed training process the dataset, and the others will use the cache
#         torch.distributed.barrier()

#     if output_examples:
#         return dataset, examples, features
#     return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_TYPES),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from huggingface.co",
    )

    parser.add_argument(
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
    )
    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")

    parser.add_argument("--block_skim", action="store_true", help="add block skim module")
    parser.add_argument("--actual_skim", action="store_true", help="perform actual skimming on input seq dim.")
    parser.add_argument("--block_size", type=int, default=32, help="block size for block skim module")
    parser.add_argument("--skim_factor", default=0.0001, type=float, help="factor for skim predictor")
    parser.add_argument("--balance_factor", default=1, type=float, help="factor for skim predictor")
    parser.add_argument("--cache_name", type=str, help="cached feature dir")
    parser.add_argument("--augment_layers", type=int, nargs="+", help="layers to augment blockskim module")
    parser.add_argument("--skim_threshold", type=float, default=0.5, help="threshold for skim predictor")

    args = parser.parse_args()

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        use_fast=False,  # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
    )
    if args.block_skim:
        if not args.model_type in ['bert', 'albert']:
            raise ValueError(f"{args.model_type} with skim not implemented")
        if args.actual_skim:
            config.actual_skim = True
        config.block_size = args.block_size
        config.augment_layers = args.augment_layers if args.augment_layers else list(range(config.num_hidden_layers))
        config.skim_threshold = args.skim_threshold
        config.output_attentions = True
        if args.model_type =='bert':
            model = BertForQuestionAnsweringWithSkim.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
        elif args.model_type == 'albert':
            model = AlbertForQuestionAnsweringWithSkim.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir if args.cache_dir else None,
            )
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        if args.block_skim:
            if args.model_type =='bert':
                model = BertForQuestionAnsweringWithSkim.from_pretrained(args.output_dir,config=config)
            elif args.model_type == 'albert':
                model = AlbertForQuestionAnsweringWithSkim.from_pretrained(args.output_dir,config=config)
        else:
            model = AutoModelForQuestionAnswering.from_pretrained(args.output_dir)  # , force_download=True)

        # SquadDataset is not compatible with Fast tokenizers which have a smarter overflow handeling
        # So we use use_fast=False here for now until Fast-tokenizer-compatible-examples are out
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case, use_fast=False)
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
                )

        else:
            logger.info("Loading checkpoint %s for evaluation", args.model_name_or_path)
            checkpoints = [args.model_name_or_path]

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            if args.block_skim:
                if args.model_type =='bert':
                    model = BertForQuestionAnsweringWithSkim.from_pretrained(checkpoint,config=config)
                elif args.model_type == 'albert':
                    model = AlbertForQuestionAnsweringWithSkim.from_pretrained(checkpoint,config=config)
            else:
                model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)  # , force_download=True)
            model.to(args.device)

            # Evaluate
            result = evaluate(args, model, tokenizer, prefix=global_step)

            result = dict((k + ("_{}".format(global_step) if global_step else ""), v) for k, v in result.items())
            results.update(result)

    logger.info("Results: {}".format(results))

    return results


if __name__ == "__main__":
    main()