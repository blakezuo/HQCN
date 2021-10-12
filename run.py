from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import json

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange

from pytorch_transformers import (WEIGHTS_NAME, BertConfig, BertTokenizer)
from Models import HQCN

from pytorch_transformers import AdamW, WarmupLinearSchedule

from utils import (compute_metrics, convert_examples_to_features, output_modes, PairHQCNDataset, HQCNDataset)
from scipy.special import softmax
from vocab import Vocab
from evaluate import Map, mrr, ndcg
import copy

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def train(args, train_dataset, eval_dataset, model, vocab, alpha = 0.8):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
                                  batch_size=args.train_batch_size, num_workers=args.num_workers)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, correct_bias=True)
    args.warmup_steps = int(t_total * args.warmup_portion)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
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
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            guids = batch['guid']
            for b, item in enumerate(batch['s_index']):
                batch['s_index'][b] = item + b * args.history_num
            batch = {k: v.to(args.device) for k, v in batch.items() if k not in ['guid','query_id', 'doc_id']}
            inputs = {'queries':      batch['queries_ids'],
                      'can_index':    batch['s_index'],
                      'wss_label':    batch['wss_label']}

            inputs['documents'] = batch['documents_pos_ids']
            inputs['features'] = batch['pos_features']
            pos_score, loss_reform = model(**inputs)
            inputs['documents'] = batch['documents_neg_ids']
            inputs['features'] = batch['neg_features']
            neg_score, _ = model(**inputs)
            # print(pos_score, neg_score)
            label = torch.ones(pos_score.size()).cuda()
            crit = nn.MarginRankingLoss(margin=1, size_average=True).cuda()
            loss = crit(pos_score, neg_score, Variable(label, requires_grad=False))
            loss = alpha * loss + (1 - alpha) * loss_reform

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            epoch_iterator.set_description('loss %s' % str(loss.item()))
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics                   
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint if it outperforms previous models
                    # Only evaluate when single GPU otherwise metrics may not average well
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)
                    model_name = os.path.join(output_dir, WEIGHTS_NAME)
                    torch.save(model.state_dict(), model_name)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, eval_dataset, model, vocab, batch_size, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task = args.task_name
    eval_output_dir = args.output_dir

    results = {}
    # eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    # eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, 
                                 batch_size=args.eval_batch_size, num_workers=args.num_workers)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    all_eval_guids = []
    all_query_ids, all_doc_ids = None, None
    wf = open(os.path.join(args.output_dir, "pred_" + prefix + ".txt"), 'w')
    maps, mrrs, ndcg1, ndcg5, ndcg10, count = 0.0,0.0,0.0,0.0,0.0, 0
    # gt = [1] + [0] * 49
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        eval_guids = batch['guid']
        # index = batch['s_index'][0]
        # if index != 5:
        #     continue
        gt = batch['label']
        index = int(eval_guids[0].split('_')[1])

        # if index >= 2:
        #     continue
        for b, item in enumerate(batch['s_index']):
            batch['s_index'][b] = item + b * args.history_num
        new_batch = {k: v.to(args.device) for k, v in batch.items() if k not in ['guid','query_id', 'doc_id']}
        
        with torch.no_grad():
            inputs = {'queries':      new_batch['queries_ids'],
                      'documents':    new_batch['documents_ids'],
                      'features':     new_batch['features'],
                      'can_index':    new_batch['s_index'],
                      'wss_label':    new_batch['wss_label']}

            pred, _ = model(**inputs)
            eval_loss = 0.0
            maps += Map(gt, pred)
            mrrs += mrr(gt, pred)
            ndcg1 += ndcg(1)(gt, pred)
            ndcg5 += ndcg(5)(gt, pred)
            ndcg10 += ndcg(10)(gt, pred)
            count += 1
            for p, l in zip(pred, gt):
                wf.write(str(p.detach().cpu().numpy()) + '\t' + str(l.detach().cpu().numpy()) + '\n')

    print('*'*100)
    print('MAP ', 1.0 * maps / count)
    print('MRR ', 1.0 * mrrs / count)
    print('NDCG@1 ', 1.0 * ndcg1 / count)
    print('NDCG@5 ', 1.0 * ndcg5 / count)
    print('NDCG@10 ', 1.0 * ndcg10 / count)
    print('*'*100)

    return


parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--data_dir", default='data/aol', type=str, required=False,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--task_name", default='stateful_search', type=str, required=False,
                    help="The name of the task to train")
parser.add_argument("--output_dir", default='output', type=str, required=False,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--embed_file", default='data/aol/fasttext.model', type=str, required=False,
                    help="embedding file")
parser.add_argument("--embed_size", default=256, type=int,
                    help="The size of word embedding")

## Other parameters
parser.add_argument("--do_train", default=True, type=str2bool,
                    help="Whether to run training.")
parser.add_argument("--do_eval", default=True, type=str2bool,
                    help="Whether to run eval on the dev set.")
parser.add_argument("--evaluate_during_training", default=True, type=str2bool,
                    help="Run evaluation during training at each logging step.")
parser.add_argument("--do_lower_case", default=True, type=str2bool,
                    help="Set this flag if you are using an uncased model.")

parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                    help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=192, type=int,
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--per_gpu_test_batch_size", default=24, type=int,
                    help="Batch size per GPU/CPU for testing.")
parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--learning_rate", default=1e-4, type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float,
                    help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float,
                    help="Max gradient norm.")
parser.add_argument("--num_train_epochs", default=3.0, type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--max_steps", default=-1, type=int,
                    help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--warmup_steps", default=0, type=int,
                    help="Linear warmup over warmup_steps.")

parser.add_argument('--logging_steps', type=int, default=5,
                    help="Log and save checkpoint every X updates steps.")
parser.add_argument('--save_steps', type=int, default=5,
                    help="Save checkpoint every X updates steps, this is disabled in our code")
parser.add_argument("--eval_all_checkpoints", default=False, type=str2bool,
                    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
parser.add_argument("--no_cuda", default=False, type=str2bool,
                    help="Avoid using CUDA when available")
parser.add_argument('--overwrite_output_dir', default=True, type=str2bool,
                    help="Overwrite the content of the output directory")
parser.add_argument('--overwrite_cache', action='store_true',
                    help="Overwrite the cached training and evaluation sets")
parser.add_argument('--seed', type=int, default=42,
                    help="random seed for initialization")

parser.add_argument('--fp16', default=False, type=str2bool,
                    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument('--fp16_opt_level', type=str, default='O1',
                    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                         "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument("--local_rank", type=int, default=-1,
                    help="For distributed training: local_rank")
parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

# parameters we added
parser.add_argument("--dataset", default='aol', type=str, required=False,
                    help="aol or msmarco. For bing data, we do not use the first query in a session")
parser.add_argument("--history_num", default=5, type=int, required=False,
                    help="number of history turns to concat")
parser.add_argument("--max_query_length", default=7, type=int, required=False,
                    help="max length of query")
parser.add_argument("--max_doc_length", default=15, type=int, required=False,
                    help="max length of document")
parser.add_argument("--inner_dim", default=200, type=int, required=False,
                    help="max length of document")
parser.add_argument("--n_layers", default=6, type=int, required=False,
                    help="layers of transformer")
parser.add_argument("--n_head", default=8, type=int, required=False,
                    help="head number")
parser.add_argument("--num_workers", default=0, type=int, required=False,
                    help="number of workers for dataloader")
parser.add_argument("--dropout", default=0.1, type=float,
                    help="dropout rate ")
parser.add_argument("--warmup_portion", default=0.1, type=float,
                    help="Linear warmup over warmup_steps (=t_total * warmup_portion). override warmup_steps ")

# args = parser.parse_args()
args, unknown = parser.parse_known_args()

if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
    raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

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
    args.n_gpu = torch.cuda.device_count()
else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    args.n_gpu = 1

args.device = device

# Setup logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

# Set seed
set_seed(args)

args.task_name = args.task_name.lower()

args.output_mode = output_modes[args.task_name]
label_list = ["False", "True"]
num_labels = len(label_list)


if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

if args.local_rank == 0:
    torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

logger.info("Training/evaluation parameters %s", args)

vocab = Vocab(args.embed_file)
args.max_seq_length = max(args.max_query_length, args.max_doc_length)
vocab.init_pretrained_embeddings(args.embed_size, args.embed_file)

d_k = args.embed_size // args.n_head
model = HQCN(vocab, d_word_vec=args.embed_size, d_model=args.embed_size, d_inner=args.inner_dim, n_layers=args.n_layers,
                    n_head=args.n_head, d_k=d_k, d_v=d_k, dropout=args.dropout, n_position = args.max_seq_length)

model.to(args.device)
# Training
if args.do_train:
    train_dataset = PairHQCNDataset(os.path.join(args.data_dir, "train_long_feature.json"), args.max_query_length, args.max_doc_length, 
                                           args.output_mode, args.dataset, args.history_num, vocab)
    eval_dataset = HQCNDataset(os.path.join(args.data_dir, "train_long_feature.json"), args.max_query_length, args.max_doc_length, 
                                           args.output_mode, args.dataset, args.history_num, vocab)
    test_dataset = HQCNDataset(os.path.join(args.data_dir, "train_long_feature.json"), args.max_query_length, args.max_doc_length, 
                                           args.output_mode, args.dataset, args.history_num, vocab)

    global_step, tr_loss = train(args, train_dataset, eval_dataset, model, vocab)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


if not args.do_train and args.do_eval:
    eval_dataset = HQCNDataset(os.path.join(args.data_dir, "test_long_feature.json"), args.max_query_length, args.max_doc_length, 
                                           args.output_mode, args.dataset, args.history_num, vocab)
    test_dataset = HQCNDataset(os.path.join(args.data_dir, "test_long_feature.json"), args.max_query_length, args.max_doc_length, 
                                           args.output_mode, args.dataset, args.history_num, vocab)

best_eval_mrr = 0.0
best_global_step = 0
results = {}
if args.do_eval and args.local_rank in [-1, 0]:
    logger.info("Eval on all checkpoints with dev set")
    # tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    checkpoints = [args.output_dir]
    checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
    logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    logger.info("Evaluate the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:
        global_step = checkpoint.split('-')[-1]
        # if global_step != '16000':
        #     continue
        print('global_step', global_step)
        # model = model_class.from_pretrained(checkpoint)

        state_dict = torch.load(os.path.join(checkpoint, WEIGHTS_NAME), map_location=args.device)
        model.load_state_dict(state_dict)
        model.to(args.device)
        # result, eval_output = evaluate(args, test_dataset, model, vocab, 
        #                                args.per_gpu_eval_batch_size, prefix=global_step)
        evaluate(args, eval_dataset, model, vocab, args.per_gpu_eval_batch_size, prefix=global_step)
