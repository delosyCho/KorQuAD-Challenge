from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
from io import open

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)
from tqdm import tqdm, trange

from modeling_bert_custom import QuestionAnswering, Config
from optimization import AdamW, WarmupLinearSchedule
from tokenization import BertTokenizer
from korquad_utils import read_squad_examples, convert_examples_to_features

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    # Required parameters

    parser.add_argument("--output_dir", default='output', type=str,
                        help="The output directory where the model checkpoints and predictions will be written.")
    parser.add_argument("--checkpoint", default='pretrain_ckpt/bert_small_ckpt.bin',
                        type=str,
                        help="checkpoint")
    parser.add_argument("--model_config", default='data/bert_small.json',
                        type=str)
    # Other parameters
    parser.add_argument("--train_file", default='data/KorQuAD_v1.0_train.json', type=str,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=96, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=8.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {} n_gpu: {}, 16-bits training: {}".format(
        device, n_gpu, args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer('data/ko_vocab_32k.txt',
                              max_len=args.max_seq_length,
                              do_basic_tokenize=True)
    # Prepare model
    config = Config.from_json_file(args.model_config)
    model = QuestionAnswering(config)
    model.bert.load_state_dict(torch.load(args.checkpoint))
    num_params = count_parameters(model)
    logger.info("Total Parameter: %d" % num_params)
    model.to(device)
    model = torch.nn.DataParallel(model)

    cached_train_features_file = args.train_file + '_{0}_{1}_{2}'.format(str(args.max_seq_length), str(args.doc_stride),
                                                                         str(args.max_query_length))
    train_examples = read_squad_examples(input_file=args.train_file, is_training=True, version_2_with_negative=False)
    try:
        with open(cached_train_features_file, "rb") as reader:
            train_features = pickle.load(reader)
    except:
        train_features = convert_examples_to_features(
            examples=train_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True)
        logger.info("  Saving train features into cached file %s", cached_train_features_file)
        with open(cached_train_features_file, "wb") as writer:
            pickle.dump(train_features, writer)

    num_train_optimization_steps = int(len(train_features) / args.train_batch_size) * args.num_train_epochs

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer,
                                     warmup_steps=num_train_optimization_steps*0.1,
                                     t_total=num_train_optimization_steps)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    logger.info("***** Running training *****")
    logger.info("  Num orig examples = %d", len(train_examples))
    logger.info("  Num split examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    num_train_step = num_train_optimization_steps

    input_ids = np.load('input_ids2.npy')
    input_mask = np.load('input_mask.npy')
    input_segments = np.load('input_segments.npy')
    start_prob = np.load('start_prob.npy')
    end_prob = np.load('end_prob.npy')
    start_label = np.load('input_start.npy')
    stop_label = np.load('input_stop.npy')
    """
    for i in range(1000):
        print(input_ids[i])
        print(max(start_prob[i]))
        print(sum(start_prob[i]))

        input()
    """
    paragraph = torch.tensor(input_ids.astype(np.int64)).type(dtype=torch.long).cuda()
    paragraph_mask = torch.tensor(input_mask.astype(np.int64)).type(dtype=torch.long).cuda()
    paragraph_segments = torch.tensor(input_segments.astype(np.int64)).type(dtype=torch.long).cuda()
    start_prob = torch.tensor(start_prob.astype(np.float32)).type(dtype=torch.float32).cuda()
    end_prob = torch.tensor(end_prob.astype(np.float32)).type(dtype=torch.float32).cuda()
    start_label = torch.tensor(start_label.astype(np.int64)).type(dtype=torch.long).cuda()
    stop_label = torch.tensor(stop_label.astype(np.int64)).type(dtype=torch.long).cuda()

    train_data = TensorDataset(paragraph, paragraph_mask, paragraph_segments, start_label, stop_label,
                               start_prob, end_prob)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    model.train()
    global_step = 0
    epoch = 0
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        iter_bar = tqdm(train_dataloader, desc="Train(XX Epoch) Step(XX/XX) (Mean loss=X.X) (loss=X.X)")
        tr_step, total_loss, mean_loss = 0, 0., 0.
        for step, batch in enumerate(iter_bar):
            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
            input_ids, input_mask, segment_ids, start_positions, end_positions, start_probs, end_probs = batch
            loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions,
                         start_probs, end_probs)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            tr_step += 1
            total_loss += loss
            mean_loss = total_loss / tr_step
            iter_bar.set_description("Train Step(%d / %d) (Mean loss=%5.5f) (loss=%5.5f)" %
                                     (global_step, num_train_step, mean_loss, loss.item()))

        logger.info("** ** * Saving file * ** **")
        model_checkpoint = "korquad_%d.bin" % (epoch)
        logger.info(model_checkpoint)
        output_model_file = os.path.join(args.output_dir,model_checkpoint)
        if n_gpu > 1:
            torch.save(model.module.state_dict(), output_model_file)
        else:
            torch.save(model.state_dict(), output_model_file)
        epoch += 1


if __name__ == "__main__":
    main()
