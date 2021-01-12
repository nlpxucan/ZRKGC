# coding=utf-8
""" Train Code """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import torch
import random
#global_seed = 42

#torch.manual_seed(global_seed)
#os.environ['PYTHONHASHSEED'] = str(global_seed)
#torch.cuda.manual_seed(global_seed)
#torch.cuda.manual_seed_all(global_seed)
#np.random.seed(global_seed)
#random.seed(global_seed)
#torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.enabled = True



import heapq
import operator

import logging
import glob
import math
import json
import argparse

import codecs
from pathlib import Path
from tqdm import tqdm, trange

from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_bert.tokenization import BertTokenizer, WhitespaceTokenizer
from modeling import BertForPreTrainingLossMask
from pytorch_bert.optimization import BertAdam, warmup_linear

from nn.data_parallel import DataParallelImbalance
import seq2seq_loader as seq2seq_loader
import torch.distributed as dist
from batch_decode import decode_batch
from metrics import f_one

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def move_stop_words(str):
	item = " ".join([w for w in str.split() if not w.lower() in stop_words])
	return item

def truncate(str, num):
	str = str.strip()
	length = len(str.split())
	list = str.split()[max(0, length - num):]
	return " ".join(list)

def detokenize(tk_str):
	tk_list = tk_str.strip().split()
	r_list = []
	for tk in tk_list:
		if tk.startswith('##') and len(r_list) > 0:
			r_list[-1] = r_list[-1] + tk[2:]
		else:
			r_list.append(tk)
	return " ".join(r_list)


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger(__name__)

def _get_max_epoch_model(output_dir):
	fn_model_list = glob.glob(os.path.join(output_dir, "model.*.bin"))
	fn_optim_list = glob.glob(os.path.join(output_dir, "optim.*.bin"))
	if (not fn_model_list) or (not fn_optim_list):
		return None
	both_set = set([int(Path(fn).stem.split('.')[-1]) for fn in fn_model_list]) & set([int(Path(fn).stem.split('.')[-1]) for fn in fn_optim_list])
	if both_set:
		return max(both_set)
	else:
		return None

def knowledge_selection(data_path, src_path, out_path):
	with open(data_path, "r", encoding="utf-8") as file:
		data = file.readlines()
	with open(src_path, "r", encoding="utf-8") as file:
		src = file.readlines()

	query_know_dict = {}
	t = 0
	for i in range(len(src)):
		query = src[i].strip().split("<#Q2K#>")[0]
		know_list = src[i].strip().split("<#Q2K#>")[1].split("<#K#>")
		for num in range(len(know_list)):
			try:
				query_know_dict[str(i) + "\t" + query].append(data[t].strip().split("<#Q2K#>")[1].strip())
				t += 1
			except:
				query_know_dict[str(i) + "\t" + query] = []
				query_know_dict[str(i) + "\t" + query].append(data[t].strip().split("<#Q2K#>")[1].strip())
				t += 1

	assert t == len(data)
	with open(out_path, "w", encoding="utf-8") as out:
		count = 0
		for query, knows in query_know_dict.items():
			check_sent = knows[0].split("\t")[0]
			random.shuffle(knows)
			bleu_list = []
			know_list = []
			for know in knows:
				know_list.append(know.split("\t")[0])
				bleu_list.append(know.split("\t")[1])
			know_bleu_map = {}
			for i in range(len(bleu_list)):
				know_bleu_map[know_list[i]] = bleu_list[i]
			sorted_knows = sorted(know_bleu_map.items(), key=operator.itemgetter(1), reverse=True)

			line = truncate(query.strip().split("\t")[1].strip(), 128)
			line += " <#Q2K#> "
			for t in range(len(sorted_knows)):
				line += sorted_knows[t][0]
				line += " <#K#> "
			line = " ".join(line.strip().split()[:-1])
			line = line.replace(" 。", "")
			line = " ".join(line.strip().split()[:210])

			if check_sent in line:
				count += 1
			out.write(line)
			out.write("\n")
		print(len(query_know_dict))
		print(count / len(query_know_dict))


def main():
	parser = argparse.ArgumentParser()

	# Required parameters
	parser.add_argument("--data_dir",
						default=None,
						type=str,
						required=True,
						help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
	#Train File
	parser.add_argument("--src_file", default=None, type=str,
						help="The input data src file name.")
	parser.add_argument("--tgt_file", default=None, type=str,
						help="The input data tgt file name.")
	parser.add_argument("--check_file", default=None, type=str,
						help="The input check knowledge data file name")

	#KS File
	parser.add_argument("--ks_src_file", default=None, type=str,
						help="The input ks data src file name.")
	parser.add_argument("--ks_tgt_file", default=None, type=str,
						help="The input ks data tgt file name.")

	parser.add_argument("--predict_input_file", default=None, type=str,
						help="predict_input_file")
	parser.add_argument("--predict_output_file", default=None, type=str,
						help="predict_output_file")

	parser.add_argument("--bert_model", default=None, type=str, required=True,
						help="Bert pre-trained model selected in the list: bert-base-uncased, "
							 "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
	parser.add_argument("--config_path", default=None, type=str,
						help="Bert config file path.")
	parser.add_argument("--output_dir",
						default=None,
						type=str,
						required=True,
						help="The output directory where the model predictions and checkpoints will be written.")
	parser.add_argument("--log_dir",
						default='',
						type=str,
						required=True,
						help="The output directory where the log will be written.")
	parser.add_argument("--model_recover_path",
						default=None,
						type=str,
						required=True,
						help="The file of fine-tuned pretraining model.")
	parser.add_argument("--optim_recover_path",
						default=None,
						type=str,
						help="The file of pretraining optimizer.")
	parser.add_argument("--predict_bleu",
						default=0.2,
						type=float,
						help="The Predicted Bleu for KS Predict ")

	# Other parameters
	parser.add_argument("--max_seq_length",
						default=128,
						type=int,
						help="The maximum total input sequence length after WordPiece tokenization. \n"
							 "Sequences longer than this will be truncated, and sequences shorter \n"
							 "than this will be padded.")
	parser.add_argument("--do_train",
						action='store_true',
						help="Whether to run training.")
	parser.add_argument("--do_predict",
						action='store_true',
						help="Whether to run ks predict.")
	parser.add_argument("--do_lower_case",
						action='store_true',
						help="Set this flag if you are using an uncased model.")
	parser.add_argument("--train_batch_size",
						default=32,
						type=int,
						help="Total batch size for training.")
	parser.add_argument("--eval_batch_size",
						default=64,
						type=int,
						help="Total batch size for eval.")
	parser.add_argument("--train_avg_bpe_length",
						default=25,
						type=int,
						help="average bpe length for train.")
	parser.add_argument("--learning_rate", default=5e-5, type=float,
						help="The initial learning rate for Adam.")
	parser.add_argument("--label_smoothing", default=0, type=float,
						help="The initial learning rate for Adam.")
	parser.add_argument("--weight_decay",
						default=0.01,
						type=float,
						help="The weight decay rate for Adam.")
	parser.add_argument("--finetune_decay",
						action='store_true',
						help="Weight decay to the original weights.")
	parser.add_argument("--num_train_epochs",
						default=3.0,
						type=float,
						help="Total number of training epochs to perform.")
	parser.add_argument("--warmup_proportion_step",
						default=300,
						type=int,
						help="Proportion of training to perform linear learning rate warmup for. ")
	parser.add_argument("--hidden_dropout_prob", default=0.1, type=float,
						help="Dropout rate for hidden states.")
	parser.add_argument("--attention_probs_dropout_prob", default=0.1, type=float,
						help="Dropout rate for attention probabilities.")
	parser.add_argument("--no_cuda",
						action='store_true',
						help="Whether not to use CUDA when available")
	parser.add_argument("--local_rank",
						type=int,
						default=-1,
						help="local_rank for distributed training on gpus")
	parser.add_argument('--seed',
						type=int,
						default=67,
						help="random seed for initialization")
	parser.add_argument('--gradient_accumulation_steps',
						type=int,
						default=1,
						help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument('--fp16', action='store_true',
						help="Whether to use 16-bit float precision instead of 32-bit")
	parser.add_argument('--fp32_embedding', action='store_true',
						help="Whether to use 32-bit float precision instead of 16-bit for embeddings")
	parser.add_argument('--loss_scale', type=float, default=0,
						help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
							 "0 (default value): dynamic loss scaling.\n"
							 "Positive power of 2: static loss scaling value.\n")
	parser.add_argument('--amp', action='store_true',
						help="Whether to use amp for fp16")
	parser.add_argument('--from_scratch', action='store_true',
						help="Initialize parameters with random values (i.e., training from scratch).")
	parser.add_argument('--new_segment_ids', action='store_true',
						help="Use new segment ids for bi-uni-directional LM.")
	parser.add_argument('--new_pos_ids', action='store_true',
						help="Use new position ids for LMs.")
	parser.add_argument('--tokenized_input', action='store_true',
						help="Whether the input is tokenized.")
	parser.add_argument('--max_len_a', type=int, default=0,
						help="Truncate_config: maximum length of segment A.")
	parser.add_argument('--max_len_b', type=int, default=0,
						help="Truncate_config: maximum length of segment B.")
	parser.add_argument('--trunc_seg', default='',
						help="Truncate_config: first truncate segment A/B (option: a, b).")
	parser.add_argument('--always_truncate_tail', action='store_true',
						help="Truncate_config: Whether we should always truncate tail.")
	parser.add_argument("--mask_prob", default=0.15, type=float,
						help="Number of prediction is sometimes less than max_pred when sequence is short.")
	parser.add_argument("--mask_prob_eos", default=0, type=float,
						help="Number of prediction is sometimes less than max_pred when sequence is short.")
	parser.add_argument('--max_pred', type=int, default=20,
						help="Max tokens of prediction.")
	parser.add_argument("--num_workers", default=0, type=int,
						help="Number of workers for the data loader.")

	parser.add_argument('--mask_source_words', action='store_true',
						help="Whether to mask source words for training")
	parser.add_argument('--skipgram_prb', type=float, default=0.0,
						help='prob of ngram mask')
	parser.add_argument('--skipgram_size', type=int, default=1,
						help='the max size of ngram mask')
	parser.add_argument('--mask_whole_word', action='store_true',
						help="Whether masking a whole word.")
	parser.add_argument('--do_l2r_training', action='store_true',
						help="Whether to do left to right training")
	parser.add_argument('--has_sentence_oracle', action='store_true',
						help="Whether to have sentence level oracle for training. "
							 "Only useful for summary generation")
	parser.add_argument('--max_position_embeddings', type=int, default=None,
						help="max position embeddings")
	parser.add_argument('--relax_projection', action='store_true',
						help="Use different projection layers for tasks.")
	parser.add_argument('--ffn_type', default=0, type=int,
						help="0: default mlp; 1: W((Wx+b) elem_prod x);")
	parser.add_argument('--num_qkv', default=0, type=int,
						help="Number of different <Q,K,V>.")
	parser.add_argument('--seg_emb', action='store_true',
						help="Using segment embedding for self-attention.")
	parser.add_argument('--s2s_special_token', action='store_true',
						help="New special tokens ([S2S_SEP]/[S2S_CLS]) of S2S.")
	parser.add_argument('--s2s_add_segment', action='store_true',
						help="Additional segmental for the encoder of S2S.")
	parser.add_argument('--s2s_share_segment', action='store_true',
						help="Sharing segment embeddings for the encoder of S2S (used with --s2s_add_segment).")
	parser.add_argument('--pos_shift', action='store_true',
						help="Using position shift for fine-tuning.")

	args = parser.parse_args()

	assert Path(args.model_recover_path).exists(), "--model_recover_path doesn't exist"

	args.output_dir = args.output_dir.replace('[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))
	args.log_dir = args.log_dir.replace('[PT_OUTPUT_DIR]', os.getenv('PT_OUTPUT_DIR', ''))

	os.makedirs(args.output_dir, exist_ok=True)
	os.makedirs(args.log_dir, exist_ok=True)

	handler = logging.FileHandler(os.path.join(args.log_dir, "train.log"), encoding='UTF-8')
	handler.setLevel(logging.INFO)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	handler.setFormatter(formatter)

	console = logging.StreamHandler()
	console.setLevel(logging.DEBUG)

	logger.addHandler(handler)
	logger.addHandler(console)

	json.dump(args.__dict__, open(os.path.join(args.output_dir, 'opt.json'), 'w'), sort_keys=True, indent=2)

	if args.local_rank == -1 or args.no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
		n_gpu = torch.cuda.device_count()
	else:
		torch.cuda.set_device(args.local_rank)
		device = torch.device("cuda", args.local_rank)
		n_gpu = 1
		# Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		dist.init_process_group(backend='nccl')
	logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(device, n_gpu, bool(args.local_rank != -1), args.fp16))

	if args.gradient_accumulation_steps < 1:
		raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
			args.gradient_accumulation_steps))

	args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
	
	#Random Seed
	
	#torch.backends.cudnn.enabled = False
	#torch.backends.cudnn.benchmark = False
	#torch.backends.cudnn.deterministic = True
	# if n_gpu > 0:
	# 	torch.cuda.manual_seed_all(args.seed)

	if args.local_rank not in (-1, 0):
		# Make sure only the first process in distributed training will download model & vocab
		dist.barrier()
	tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
	if args.max_position_embeddings:
		tokenizer.max_len = args.max_position_embeddings
	data_tokenizer = WhitespaceTokenizer() if args.tokenized_input else tokenizer
	if args.local_rank == 0:
		dist.barrier()

	#Data process pipelines
	bi_uni_pipeline = [seq2seq_loader.Preprocess4Seq2seq(args.max_pred, args.mask_prob, list(tokenizer.vocab.keys(
	)), tokenizer.convert_tokens_to_ids, args.max_seq_length, new_segment_ids=args.new_segment_ids,
														 truncate_config={'max_len_a': args.max_len_a,
																		  'max_len_b': args.max_len_b,
																		  'trunc_seg': args.trunc_seg,
																		  'always_truncate_tail': args.always_truncate_tail},
														 mask_source_words=args.mask_source_words,
														 skipgram_prb=args.skipgram_prb,
														 skipgram_size=args.skipgram_size,
														 mask_whole_word=args.mask_whole_word, mode="s2s",
														 has_oracle=args.has_sentence_oracle, num_qkv=args.num_qkv,
														 s2s_special_token=args.s2s_special_token,
														 s2s_add_segment=args.s2s_add_segment,
														 s2s_share_segment=args.s2s_share_segment,
														 pos_shift=args.pos_shift)]
	C_bi_uni_pipeline = [seq2seq_loader.C_Preprocess4Seq2seq(args.max_pred, args.mask_prob, list(tokenizer.vocab.keys(
	)), tokenizer.convert_tokens_to_ids, args.max_seq_length, new_segment_ids=args.new_segment_ids,
															 truncate_config={'max_len_a': args.max_len_a,
																			  'max_len_b': args.max_len_b,
																			  'trunc_seg': args.trunc_seg,
																			  'always_truncate_tail': args.always_truncate_tail},
															 mask_source_words=args.mask_source_words,
															 skipgram_prb=args.skipgram_prb,
															 skipgram_size=args.skipgram_size,
															 mask_whole_word=args.mask_whole_word, mode="s2s",
															 has_oracle=args.has_sentence_oracle, num_qkv=args.num_qkv,
															 s2s_special_token=args.s2s_special_token,
															 s2s_add_segment=args.s2s_add_segment,
															 s2s_share_segment=args.s2s_share_segment,
															 pos_shift=args.pos_shift)]
	ks_predict_bi_uni_pipeline = [
		seq2seq_loader.Preprocess4Seq2seq_predict(args.max_pred, args.mask_prob,
												  list(tokenizer.vocab.keys(
												  )), tokenizer.convert_tokens_to_ids,
												  args.max_seq_length,
												  new_segment_ids=args.new_segment_ids,
												  truncate_config={'max_len_a': args.max_len_a,
																   'max_len_b': args.max_len_b,
																   'trunc_seg': args.trunc_seg,
																   'always_truncate_tail': args.always_truncate_tail},
												  mask_source_words=args.mask_source_words,
												  skipgram_prb=args.skipgram_prb,
												  skipgram_size=args.skipgram_size,
												  mask_whole_word=args.mask_whole_word, mode="s2s",
												  has_oracle=args.has_sentence_oracle,
												  num_qkv=args.num_qkv,
												  s2s_special_token=args.s2s_special_token,
												  s2s_add_segment=args.s2s_add_segment,
												  s2s_share_segment=args.s2s_share_segment,
												  pos_shift=args.pos_shift)]

	if args.do_train:
		print("Loading QKR Train Dataset", args.data_dir)
		file_oracle = None
		if args.has_sentence_oracle:
			file_oracle = os.path.join(args.data_dir, 'train.oracle')
		fn_src = os.path.join(args.data_dir, args.src_file if args.src_file else 'train.src')
		fn_tgt = os.path.join(args.data_dir, args.tgt_file if args.tgt_file else 'train.tgt')
		fn_check = os.path.join(args.data_dir, args.check_file)

		train_dataset = seq2seq_loader.C_Seq2SeqDataset(fn_src, fn_tgt, fn_check, args.train_batch_size, data_tokenizer, args.max_seq_length,
														file_oracle=file_oracle, bi_uni_pipeline=C_bi_uni_pipeline)
		if args.local_rank == -1:
			train_sampler = RandomSampler(train_dataset, replacement=False)
			_batch_size = args.train_batch_size
		else:
			train_sampler = DistributedSampler(train_dataset)
			_batch_size = args.train_batch_size // dist.get_world_size()
		train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=_batch_size, sampler=train_sampler, num_workers=args.num_workers,
													   collate_fn=seq2seq_loader.batch_list_to_batch_tensors, pin_memory=False)

		print("Loading KS Train Dataset", args.data_dir)
		ks_fn_src = os.path.join(args.data_dir, args.ks_src_file)
		ks_fn_tgt = os.path.join(args.data_dir, args.ks_tgt_file)
		ks_train_dataset = seq2seq_loader.Seq2SeqDataset(ks_fn_src, ks_fn_tgt, args.train_batch_size, data_tokenizer, args.max_seq_length, file_oracle=file_oracle,
			bi_uni_pipeline=bi_uni_pipeline)
		if args.local_rank == -1:
			ks_train_sampler = RandomSampler(ks_train_dataset, replacement=False)
			_batch_size = args.train_batch_size
		else:
			ks_train_sampler = DistributedSampler(ks_train_dataset)
			_batch_size = args.train_batch_size // dist.get_world_size()
		ks_train_dataloader = torch.utils.data.DataLoader(ks_train_dataset, batch_size=_batch_size, sampler=ks_train_sampler, num_workers=args.num_workers,
														  collate_fn=seq2seq_loader.batch_list_to_batch_tensors, pin_memory=False)

		# note: args.train_batch_size has been changed to (/= args.gradient_accumulation_steps)
		t_total = int(len(train_dataloader) * args.num_train_epochs / args.gradient_accumulation_steps)

	amp_handle = None
	if args.fp16 and args.amp:
		from apex import amp
		amp_handle = amp.init(enable_caching=True)
		logger.info("enable fp16 with amp")

	# Prepare model
	cls_num_labels = 2
	type_vocab_size = 6 + (1 if args.s2s_add_segment else 0) if args.new_segment_ids else 2
	num_sentlvl_labels = 2 if args.has_sentence_oracle else 0
	relax_projection = 4 if args.relax_projection else 0
	if args.local_rank not in (-1, 0):
		# Make sure only the first process in distributed training will download model & vocab
		dist.barrier()

	#Recover model
	if args.model_recover_path:
		logger.info(" ** ** * Recover model: %s ** ** * ", args.model_recover_path)
		model_recover = torch.load(args.model_recover_path, map_location='cpu')
		global_step = 0

	mask_word_id, eos_word_ids, sos_word_id = tokenizer.convert_tokens_to_ids(["[MASK]", "[SEP]", "[S2S_SOS]"])

	model = BertForPreTrainingLossMask.from_pretrained(
		args.bert_model, state_dict=model_recover, num_labels=cls_num_labels, num_rel=0,
		type_vocab_size=type_vocab_size, config_path=args.config_path, task_idx=3,
		num_sentlvl_labels=num_sentlvl_labels, max_position_embeddings=args.max_position_embeddings,
		label_smoothing=args.label_smoothing, fp32_embedding=args.fp32_embedding, relax_projection=relax_projection,
		new_pos_ids=args.new_pos_ids, ffn_type=args.ffn_type, hidden_dropout_prob=args.hidden_dropout_prob,
		attention_probs_dropout_prob=args.attention_probs_dropout_prob, num_qkv=args.num_qkv, seg_emb=args.seg_emb,
		mask_word_id=mask_word_id, search_beam_size=5,
		length_penalty=0, eos_id=eos_word_ids, sos_id=sos_word_id, forbid_duplicate_ngrams=True,
		forbid_ignore_set=None, mode="s2s")

	if args.local_rank == 0:
		dist.barrier()

	if args.fp16:
		model.half()
		if args.fp32_embedding:
			model.bert.embeddings.word_embeddings.float()
			model.bert.embeddings.position_embeddings.float()
			model.bert.embeddings.token_type_embeddings.float()
	model.to(device)

	model.tmp_bert_emb.word_embeddings.weight = torch.nn.Parameter(model.bert.embeddings.word_embeddings.weight.clone())
	model.tmp_bert_emb.token_type_embeddings.weight = torch.nn.Parameter(model.bert.embeddings.token_type_embeddings.weight.clone())
	model.tmp_bert_emb.position_embeddings.weight = torch.nn.Parameter(model.bert.embeddings.position_embeddings.weight.clone())
	model.mul_bert_emb.word_embeddings.weight = torch.nn.Parameter(model.bert.embeddings.word_embeddings.weight.clone())
	model.mul_bert_emb.token_type_embeddings.weight = torch.nn.Parameter(model.bert.embeddings.token_type_embeddings.weight.clone())
	model.mul_bert_emb.position_embeddings.weight = torch.nn.Parameter(model.bert.embeddings.position_embeddings.weight.clone())
	if args.local_rank != -1:
		try:
			from torch.nn.parallel import DistributedDataParallel as DDP
		except ImportError:
			raise ImportError("DistributedDataParallel")
		model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
	elif n_gpu > 1:
		model = DataParallelImbalance(model)

	# Prepare optimizer
	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(
			nd in n for nd in no_decay)], 'weight_decay': 0.01},
		{'params': [p for n, p in param_optimizer if any(
			nd in n for nd in no_decay)], 'weight_decay': 0.0}
	]
	if args.fp16:
		try:
			from pytorch_bert.optimization_fp16 import FP16_Optimizer_State
			from apex.optimizers import FusedAdam
		except ImportError:
			raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

		optimizer = FusedAdam(optimizer_grouped_parameters, lr=args.learning_rate, bias_correction=False, max_grad_norm=1.0)
		if args.loss_scale == 0:
			optimizer = FP16_Optimizer_State(optimizer, dynamic_loss_scale=True)
		else:
			optimizer = FP16_Optimizer_State(optimizer, static_loss_scale=args.loss_scale)
	else:
		optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=args.warmup_proportion, t_total=t_total)

	if args.optim_recover_path is not None:
		logger.info(" ** ** * Recover optimizer from : {} ** ** * ".format(args.optim_recover_path))
		optim_recover = torch.load(args.optim_recover_path, map_location='cpu')
		if hasattr(optim_recover, 'state_dict'):
			optim_recover = optim_recover.state_dict()
		optimizer.load_state_dict(optim_recover)
		if args.loss_scale == 0:
			logger.info(" ** ** * Recover optimizer: dynamic_loss_scale ** ** * ")
			optimizer.dynamic_loss_scale = True

	#logger.info(" ** ** * CUDA.empty_cache() ** ** * ")
	torch.cuda.empty_cache()

	# ################# TRAIN ############################ #
	if args.do_train:
		max_F1 = 0
		best_step = 0
		logger.info(" ** ** * Running training ** ** * ")
		logger.info("  Batch size = %d", args.train_batch_size)
		logger.info("  Num steps = %d", t_total)

		model.train()
		start_epoch = 1

		for i_epoch in trange(start_epoch, start_epoch + 1, desc="Epoch", disable=args.local_rank not in (-1, 0)):
			if args.local_rank != -1:
				train_sampler.set_epoch(i_epoch)

			step = 0
			for batch, ks_batch in zip(train_dataloader, ks_train_dataloader):
				# ################# E step + M step + Mutual Information Loss ############################ #
				batch = [t.to(device) if t is not None else None for t in batch]

				input_ids, segment_ids, input_mask, mask_qkv, lm_label_ids, masked_pos, masked_weights, is_next, task_idx, tgt_pos, labels, ks_labels, check_ids = batch
				oracle_pos, oracle_weights, oracle_labels = None, None, None


				loss_tuple = model(input_ids, segment_ids, input_mask, lm_label_ids, is_next, masked_pos=masked_pos,
								   masked_weights=masked_weights, task_idx=task_idx, masked_pos_2=oracle_pos,
								   masked_weights_2=oracle_weights, masked_labels_2=oracle_labels, mask_qkv=mask_qkv, tgt_pos=tgt_pos, labels=labels.half(),
								   ks_labels=ks_labels, check_ids=check_ids)



				masked_lm_loss, next_sentence_loss, KL_loss, Mutual_loss, Golden_loss, predict_kl_loss = loss_tuple
				if n_gpu > 1:  # mean() to average on multi-gpu.
					masked_lm_loss = masked_lm_loss.mean()
					next_sentence_loss = next_sentence_loss.mean()
					Mutual_loss = Mutual_loss.mean()
					Golden_loss = Golden_loss.mean()
					KL_loss = KL_loss.mean()
					predict_kl_loss = predict_kl_loss.mean()

				loss = masked_lm_loss + next_sentence_loss + KL_loss + predict_kl_loss + Mutual_loss + Golden_loss
				logger.info("In{}step, masked_lm_loss:{}".format(step, masked_lm_loss))
				logger.info("In{}step, KL_loss:{}".format(step, KL_loss))
				logger.info("In{}step, Mutual_loss:{}".format(step, Mutual_loss))
				logger.info("In{}step, Golden_loss:{}".format(step, Golden_loss))
				logger.info("In{}step, predict_kl_loss:{}".format(step, predict_kl_loss))

				logger.info("******************************************* ")

				# ensure that accumlated gradients are normalized
				if args.gradient_accumulation_steps > 1:
					loss = loss / args.gradient_accumulation_steps

				if args.fp16:
					optimizer.backward(loss)
					if amp_handle:
						amp_handle._clear_cache()
				else:
					loss.backward()
				if (step + 1) % args.gradient_accumulation_steps == 0:
					lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion_step / t_total)
					if args.fp16:
						# modify learning rate with special warm up BERT uses
						for param_group in optimizer.param_groups:
							param_group['lr'] = lr_this_step
					optimizer.step()
					optimizer.zero_grad()
					global_step += 1

				# ################# Knowledge Selection Loss ############################ #
				if random.randint(0, 4) == 0 :
					ks_batch = [t.to(device) if t is not None else None for t in ks_batch]

					input_ids, segment_ids, input_mask, mask_qkv, lm_label_ids, masked_pos, masked_weights, is_next, task_idx, _, labels, ks_labels = ks_batch
					oracle_pos, oracle_weights, oracle_labels = None, None, None
					loss_tuple = model(input_ids, segment_ids, input_mask, lm_label_ids, is_next, masked_pos=masked_pos,
									   masked_weights=masked_weights, task_idx=task_idx, masked_pos_2=oracle_pos,
									   masked_weights_2=oracle_weights, masked_labels_2=oracle_labels, mask_qkv=mask_qkv, labels=labels,
									   ks_labels=ks_labels, train_ks=True)

					ks_loss, _ = loss_tuple
					if n_gpu > 1:  # mean() to average on multi-gpu.
						ks_loss = ks_loss.mean()
					loss = ks_loss

					logger.info("In{}step, ks_loss:{}".format(step, ks_loss))
					logger.info("******************************************* ")

					# ensure that accumlated gradients are normalized
					if args.gradient_accumulation_steps > 1:
						loss = loss / args.gradient_accumulation_steps

					if args.fp16:
						optimizer.backward(loss)
						if amp_handle:
							amp_handle._clear_cache()
					else:
						loss.backward()
					if (step + 1) % args.gradient_accumulation_steps == 0:
						lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion_step / t_total)
						if args.fp16:
							# modify learning rate with special warm up BERT uses
							for param_group in optimizer.param_groups:
								param_group['lr'] = lr_this_step
						optimizer.step()
						optimizer.zero_grad()

				step += 1
				###################### Eval Every 5000 Step ############################ #
				if (global_step + 1) % 5000 == 0:
					next_i = 0
					model.eval()

					# Know Rank Stage
					logger.info(" ** ** * DEV Know Selection Begin ** ** * ")
					with open(os.path.join(args.data_dir, args.predict_input_file), "r", encoding="utf-8") as file:
						src_file = file.readlines()
					with open(os.path.join(args.data_dir, "train_tgt_pad.empty"), "r", encoding="utf-8") as file:
						tgt_file = file.readlines()
					with open(os.path.join(args.data_dir, args.predict_output_file), "w", encoding="utf-8") as out:
						while next_i < len(src_file):
							batch_src = src_file[next_i:next_i + args.eval_batch_size]
							batch_tgt = tgt_file[next_i:next_i + args.eval_batch_size]

							next_i += args.eval_batch_size

							ex_list = []
							for src, tgt in zip(batch_src, batch_tgt):
								src_tk = data_tokenizer.tokenize(src.strip())
								tgt_tk = data_tokenizer.tokenize(tgt.strip())
								ex_list.append((src_tk, tgt_tk))

							batch = []
							for idx in range(len(ex_list)):
								instance = ex_list[idx]
								for proc in ks_predict_bi_uni_pipeline:
									instance = proc(instance)
									batch.append(instance)

							batch_tensor = seq2seq_loader.batch_list_to_batch_tensors(batch)
							batch = [t.to(device) if t is not None else None for t in batch_tensor]

							input_ids, segment_ids, input_mask, mask_qkv, lm_label_ids, masked_pos, masked_weights, is_next, task_idx = batch

							predict_bleu = args.predict_bleu * torch.ones([input_ids.shape[0]],device=input_ids.device)
							oracle_pos, oracle_weights, oracle_labels = None, None, None
							with torch.no_grad():
								logits = model(input_ids, segment_ids, input_mask, lm_label_ids, is_next,
											   masked_pos=masked_pos, masked_weights=masked_weights, task_idx=task_idx,
											   masked_pos_2=oracle_pos, masked_weights_2=oracle_weights,
											   masked_labels_2=oracle_labels, mask_qkv=mask_qkv, labels=predict_bleu, train_ks=True)

								logits = torch.nn.functional.softmax(logits, dim=1)
								labels = logits[:, 1].cpu().numpy()
								for i in range(len(labels)):
									line = batch_src[i].strip()
									line += "\t"
									line += str(labels[i])
									out.write(line)
									out.write("\n")

					data_path = os.path.join(args.data_dir, "qkr_dev.ks_score.tk")
					src_path = os.path.join(args.data_dir, "qkr_dev.src.tk")
					src_out_path = os.path.join(args.data_dir, "rank_qkr_dev.src.tk")
					tgt_path = os.path.join(args.data_dir, "qkr_dev.tgt")

					knowledge_selection(data_path, src_path, src_out_path)
					logger.info(" ** ** * DEV Know Selection End ** ** * ")

					# Decode Stage
					logger.info(" ** ** * Dev Decode Begin ** ** * ")
					with open(src_out_path, encoding="utf-8") as file:
						dev_src_lines = file.readlines()
					with open(tgt_path,encoding="utf-8") as file:
						golden_response_lines = file.readlines()

					decode_result = decode_batch(model, dev_src_lines)
					logger.info(" ** ** * Dev Decode End ** ** * ")

					# Compute dev F1
					assert len(decode_result) == len(golden_response_lines)
					C_F1 = f_one(decode_result, golden_response_lines)[0]
					logger.info("** ** * Current F1 is {} ** ** * ".format(C_F1))
					if C_F1 < max_F1:
						logger.info("** ** * Current F1 is lower than Previous F1. So Stop Training ** ** * ")
						logger.info("** ** * The best model is {} ** ** * ".format(best_step))
						break
					else:
						max_F1 = C_F1
						best_step = step
						logger.info("** ** * Current F1 is larger than Previous F1. So Continue Training ** ** * ")

					# Save trained model
					if (args.local_rank == -1 or torch.distributed.get_rank() == 0):
						logger.info("** ** * Saving fine-tuned model and optimizer ** ** * ")
						model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
						output_model_file = os.path.join(args.output_dir, "model.{}_{}.bin".format(i_epoch, global_step))
						torch.save(model_to_save.state_dict(), output_model_file)
						output_optim_file = os.path.join(args.output_dir, "optim.bin")
						torch.save(optimizer.state_dict(), output_optim_file)

						#logger.info(" ** ** * CUDA.empty_cache() ** ** * ")
						torch.cuda.empty_cache()

	# ################# Predict ############################ #
	if args.do_predict:
		bi_uni_pipeline = [
			seq2seq_loader.Preprocess4Seq2seq_predict(args.max_pred, args.mask_prob, list(tokenizer.vocab.keys(
			)), tokenizer.convert_tokens_to_ids, args.max_seq_length, new_segment_ids=args.new_segment_ids,
													  truncate_config={'max_len_a': args.max_len_a,
																	   'max_len_b': args.max_len_b,
																	   'trunc_seg': args.trunc_seg,
																	   'always_truncate_tail': args.always_truncate_tail},
													  mask_source_words=args.mask_source_words,
													  skipgram_prb=args.skipgram_prb, skipgram_size=args.skipgram_size,
													  mask_whole_word=args.mask_whole_word, mode="s2s",
													  has_oracle=args.has_sentence_oracle, num_qkv=args.num_qkv,
													  s2s_special_token=args.s2s_special_token,
													  s2s_add_segment=args.s2s_add_segment,
													  s2s_share_segment=args.s2s_share_segment,
													  pos_shift=args.pos_shift)]

		next_i = 0
		model.eval()

		with open(os.path.join(args.data_dir, args.predict_input_file), "r", encoding="utf-8") as file:
			src_file = file.readlines()
		with open("train_tgt_pad.empty", "r", encoding="utf-8") as file:
			tgt_file = file.readlines()
		with open(os.path.join(args.data_dir, args.predict_output_file), "w", encoding="utf-8") as out:
			logger.info("** ** * Continue knowledge ranking ** ** * ")
			for next_i in tqdm(range(len(src_file) // args.eval_batch_size + 1)):
			#while next_i < len(src_file):
				batch_src = src_file[next_i*args.eval_batch_size:(next_i + 1) * args.eval_batch_size]
				batch_tgt = tgt_file[next_i*args.eval_batch_size:(next_i + 1) * args.eval_batch_size]
				#next_i += args.eval_batch_size

				ex_list = []
				for src, tgt in zip(batch_src, batch_tgt):
					src_tk = data_tokenizer.tokenize(src.strip())
					tgt_tk = data_tokenizer.tokenize(tgt.strip())
					ex_list.append((src_tk, tgt_tk))

				batch = []
				for idx in range(len(ex_list)):
					instance = ex_list[idx]
					for proc in bi_uni_pipeline:
						instance = proc(instance)
						batch.append(instance)

				batch_tensor = seq2seq_loader.batch_list_to_batch_tensors(batch)
				batch = [t.to(device) if t is not None else None for t in batch_tensor]

				input_ids, segment_ids, input_mask, mask_qkv, lm_label_ids, masked_pos, masked_weights, is_next, task_idx = batch

				predict_bleu = args.predict_bleu * torch.ones([input_ids.shape[0]], device=input_ids.device)
				oracle_pos, oracle_weights, oracle_labels = None, None, None
				with torch.no_grad():
					logits = model(input_ids, segment_ids, input_mask, lm_label_ids, is_next,
								   masked_pos=masked_pos, masked_weights=masked_weights, task_idx=task_idx,
								   masked_pos_2=oracle_pos, masked_weights_2=oracle_weights,
								   masked_labels_2=oracle_labels, mask_qkv=mask_qkv, labels=predict_bleu, train_ks=True)

					logits = torch.nn.functional.softmax(logits, dim=1)
					labels = logits[:, 1].cpu().numpy()
					for i in range(len(labels)):
						line = batch_src[i].strip()
						line += "\t"
						line += str(labels[i])
						out.write(line)
						out.write("\n")


if __name__ == "__main__":
	main()
