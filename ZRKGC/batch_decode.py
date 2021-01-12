# coding=utf-8
""" Decode During Training """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import glob
import argparse
import math
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
import random
import pickle
from pathlib import Path

from pytorch_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from pytorch_bert.optimization import BertAdam, warmup_linear

from nn.data_parallel import DataParallelImbalance
import seq2seq_loader as seq2seq_loader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)


def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list

def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)


#parameter
bert_model = "unilm_v2_bert_pretrain"
max_seq_length = 256
ffn_type=0
num_qkv=0
seed=123
beam_size=5
length_penalty=0
forbid_ignore_word=None
max_tgt_length=40
batch_size = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if n_gpu > 0:
    torch.cuda.manual_seed_all(seed)

tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False)
data_tokenizer = WhitespaceTokenizer()
tokenizer.max_len = max_seq_length
pair_num_relation = 0
bi_uni_pipeline = []
bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqDecoder(list(tokenizer.vocab.keys()), tokenizer.convert_tokens_to_ids, max_seq_length, max_tgt_length=max_tgt_length,
                                                                mode="s2s", num_qkv=num_qkv, s2s_special_token=False, s2s_add_segment=False, s2s_share_segment=False, pos_shift=False))
amp_handle = None
from apex import amp
amp_handle = amp.init(enable_caching=True)
logger.info("enable fp16 with amp")

cls_num_labels = 2
type_vocab_size = 6
mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(
    ["[MASK]", "[SEP]", "[S2S_SOS]"])
forbid_ignore_set = None


def decode_batch(model,big_batch_data):

    torch.cuda.empty_cache()
    model.eval()

    max_src_length = max_seq_length - 2 - max_tgt_length
    input_lines = [x.strip() for x in big_batch_data]


    all_input_lines = [data_tokenizer.tokenize(
        x)[:max_src_length] for x in input_lines]

    total_length = len(input_lines)
    total_iter = total_length//batch_size
    if total_iter * batch_size < total_length:
        total_iter += 1

    all_output_lines = []

    for cur_iter in range(total_iter):
        input_lines = all_input_lines[cur_iter*batch_size:(cur_iter+1)*batch_size]
        input_lines = sorted(list(enumerate(input_lines)), key=lambda x: -len(x[1]))
        output_lines = [""] * len(input_lines)
        _chunk = input_lines
        buf_id = [x[0] for x in _chunk]
        buf = [x[1] for x in _chunk]

        max_a_len = max([len(x) for x in buf])
        instances = []
        for instance in [(x, max_a_len) for x in buf]:
            for proc in bi_uni_pipeline:
                instances.append(proc(instance))
        with torch.no_grad():
            batch = seq2seq_loader.batch_list_to_batch_tensors(
                instances)
            batch = [
                t.to(device) if t is not None else None for t in batch]
            input_ids, token_type_ids, position_ids, input_mask, mask_qkv, task_idx = batch
            traces = model(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids, attention_mask = input_mask, task_idx=task_idx, mask_qkv=mask_qkv, decode=True)
            if beam_size > 1:
                traces = {k: v.tolist() for k, v in traces.items()}
                output_ids = traces['pred_seq']
            else:
                output_ids = traces.tolist()
            for i in range(len(buf)):
                w_ids = output_ids[i]
                output_buf = tokenizer.convert_ids_to_tokens(w_ids)
                output_tokens = []
                for t in output_buf:
                    if t in ("[SEP]", "[PAD]"):
                        break
                    output_tokens.append(t)
                output_sequence = ' '.join(detokenize(output_tokens))
                output_lines[buf_id[i]] = output_sequence

        all_output_lines.extend(output_lines)

    assert len(all_output_lines) == len(all_input_lines)
    return all_output_lines



