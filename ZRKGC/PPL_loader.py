from random import randint, shuffle, choice
from random import random as rand
import math
import torch

from loader_utils import get_random_word, batch_list_to_batch_tensors, Pipeline
TopK = 1

# Input file format :
# 1. One sentence per line. These should ideally be actual sentences,
#    not entire paragraphs or arbitrary spans of text. (Because we use
#    the sentence boundaries for the "next sentence prediction" task).
# 2. Blank lines between documents. Document boundaries are needed
#    so that the "next sentence prediction" task doesn't span between documents.


def truncate_tokens_pair(tokens_a, tokens_b, max_len, max_len_a=0, max_len_b=0, trunc_seg=None,
						 always_truncate_tail=False):
	num_truncated_a = [0, 0]
	num_truncated_b = [0, 0]
	while True:
		if len(tokens_a) + len(tokens_b) <= max_len:
			break
		if (max_len_a > 0) and len(tokens_a) > max_len_a:
			trunc_tokens = tokens_a
			num_truncated = num_truncated_a
		elif (max_len_b > 0) and len(tokens_b) > max_len_b:
			trunc_tokens = tokens_b
			num_truncated = num_truncated_b
		elif trunc_seg:
			# truncate the specified segment
			if trunc_seg == 'a':
				trunc_tokens = tokens_a
				num_truncated = num_truncated_a
			else:
				trunc_tokens = tokens_b
				num_truncated = num_truncated_b
		else:
			# truncate the longer segment
			if len(tokens_a) > len(tokens_b):
				trunc_tokens = tokens_a
				num_truncated = num_truncated_a
			else:
				trunc_tokens = tokens_b
				num_truncated = num_truncated_b
		# whether always truncate source sequences
		if (not always_truncate_tail) and (rand() < 0.5):
			del trunc_tokens[0]
			num_truncated[0] += 1
		else:
			trunc_tokens.pop()
			num_truncated[1] += 1
	return num_truncated_a, num_truncated_b


class C_Seq2SeqDataset(torch.utils.data.Dataset):
	""" Load sentence pair (sequential or random order) from corpus """

	def __init__(self, file_src, file_tgt, batch_size, tokenizer, max_len, file_oracle=None, short_sampling_prob=0.1,
				 sent_reverse_order=False, bi_uni_pipeline=[]):
		super().__init__()
		self.tokenizer = tokenizer  # tokenize function
		self.max_len = max_len  # maximum length of tokens
		self.short_sampling_prob = short_sampling_prob
		self.bi_uni_pipeline = bi_uni_pipeline
		self.batch_size = batch_size
		self.sent_reverse_order = sent_reverse_order

		# read the file into memory
		self.ex_list = []
		if file_oracle is None:
			with open(file_src, "r", encoding='utf-8') as f_src, open(file_tgt, "r", encoding='utf-8') as f_tgt:

				f_check = "."
				f_style = ".\t0"

				for src, tgt in zip(f_src, f_tgt):
					src = src.split("[SEP]")
					tgt = tgt.split("[SEP]")
					style = f_style.split("[SEP]")
					check = f_check.split("[SEP]")
					

					src_tk = tokenizer.tokenize(src[0].strip())
					tgt_tk = tokenizer.tokenize(tgt[0].strip())
					check_tk = tokenizer.tokenize(check[0].strip())
					style_tk = tokenizer.tokenize(style[0].strip())
					assert len(src_tk) > 0
					assert len(tgt_tk) > 0
					assert len(check_tk) > 0
					assert len(style_tk) > 0

					for t in range(len(tgt_tk)):
						src_tk_list = []
						tgt_tk_list = []
						check_tk_list = []
						style_tk_list = []

						src_tk_list.append(src_tk)
						tgt_tk_list.append(tgt_tk[:t+1])
						check_tk_list.append(check_tk)
						style_tk_list.append(style_tk)

						self.ex_list.append((src_tk_list, tgt_tk_list, check_tk_list, style_tk_list))
		print('Load {0} documents'.format(len(self.ex_list)))

	def __len__(self):
		return len(self.ex_list)

	def __getitem__(self, idx):
		instance = self.ex_list[idx]
		proc = choice(self.bi_uni_pipeline)
		instance = proc(instance)
		return instance

	def __iter__(self):  # iterator to load data
		for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
			batch = []
			for __ in range(self.batch_size):
				idx = randint(0, len(self.ex_list) - 1)
				batch.append(self.__getitem__(idx))
			# To Tensor
			yield batch_list_to_batch_tensors(batch)


class C_Preprocess4Seq2seq(Pipeline):
	""" Pre-processing steps for pretraining transformer """

	def __init__(self, max_pred, mask_prob, vocab_words, indexer, max_len=512, skipgram_prb=0, skipgram_size=0,
				 block_mask=False, mask_whole_word=False, new_segment_ids=False, truncate_config={},
				 mask_source_words=False, mode="s2s", has_oracle=False, num_qkv=0, s2s_special_token=False,
				 s2s_add_segment=False, s2s_share_segment=False, pos_shift=False):
		super().__init__()
		self.max_len = max_len
		self.max_pred = max_pred  # max tokens of prediction
		self.mask_prob = mask_prob  # masking probability
		self.vocab_words = vocab_words  # vocabulary (sub)words
		self.indexer = indexer  # function from token to token index
		self.max_len = max_len
		self._tril_matrix = torch.tril(torch.ones(
			(max_len, max_len), dtype=torch.long))
		self.skipgram_prb = skipgram_prb
		self.skipgram_size = skipgram_size
		self.mask_whole_word = mask_whole_word
		self.new_segment_ids = new_segment_ids
		self.always_truncate_tail = truncate_config.get(
			'always_truncate_tail', False)
		self.max_len_a = truncate_config.get('max_len_a', None)
		self.max_len_b = truncate_config.get('max_len_b', None)
		self.trunc_seg = truncate_config.get('trunc_seg', None)
		self.task_idx = 3  # relax projection layer for different tasks
		self.mask_source_words = mask_source_words
		assert mode in ("s2s", "l2r")
		self.mode = mode
		self.has_oracle = has_oracle
		self.num_qkv = num_qkv
		self.s2s_special_token = s2s_special_token
		self.s2s_add_segment = s2s_add_segment
		self.s2s_share_segment = s2s_share_segment
		self.pos_shift = pos_shift

	def __call__(self, instance):
		input_ids_list = []
		segment_ids_list = []
		input_mask_list = []
		masked_ids_list = []
		masked_pos_list = []
		masked_weights_list = []
		tgt_pos_list = []
		labels_list = []
		ks_labels_list = []
		style_ids_list = []
		style_labels_list = []
		check_ids_list = []

		tokens_a_list, tokens_b_list, check_list, tokens_c_list = instance[:4]


		for rank in range(TopK):

			tokens_a = tokens_a_list[rank]
			tokens_b = tokens_b_list[rank]
			tokens_c = tokens_c_list[rank]
			check_tokens = check_list[rank][:self.max_pred]

			#######
			check_ids = self.indexer(check_tokens)
			# Zero Padding
			check_n_pad = self.max_pred - len(check_ids)
			check_ids.extend([0] * check_n_pad)
			assert len(check_ids) == self.max_pred
			########

			tokens_a = ["."] + tokens_a[:-1]

			labels = torch.tensor(0.1)
			ks_labels = torch.tensor(1)
			tokens_b = tokens_b

			style_labels = torch.tensor(int(tokens_c[-1]))
			tokens_c = tokens_c[:-1]

			if self.pos_shift:
				tokens_b = ['[S2S_SOS]'] + tokens_b

			# -3  for special tokens [CLS], [SEP], [SEP]
			num_truncated_a, _ = truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3, max_len_a=self.max_len_a,
													  max_len_b=self.max_len_b, trunc_seg=self.trunc_seg,
													  always_truncate_tail=self.always_truncate_tail)

			#process tokens_a all_len ==  213; tokens_b max len = 40
			tokens_a = tokens_a[:213]
			while len(tokens_a) < 213:
				tokens_a.extend(["[PAD]"])
			tokens_b = tokens_b[:40]


			# Add Special Tokens
			if self.s2s_special_token:
				tokens = ['[S2S_CLS]'] + tokens_a + \
						 ['[S2S_SEP]'] + tokens_b + ['[SEP]']
			else:
				tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

			if self.new_segment_ids:
				if self.mode == "s2s":
					if self.s2s_add_segment:
						if self.s2s_share_segment:
							segment_ids = [0] + [1] * \
										  (len(tokens_a) + 1) + [5] * (len(tokens_b) + 1)
						else:
							segment_ids = [4] + [6] * \
										  (len(tokens_a) + 1) + [5] * (len(tokens_b) + 1)
					else:
						segment_ids = [4] * (len(tokens_a) + 2) + \
									  [5] * (len(tokens_b) + 1)
				else:
					segment_ids = [2] * (len(tokens))
			else:
				segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)

			if self.pos_shift:
				n_pred = min(self.max_pred, len(tokens_b))
				masked_pos = [len(tokens_a) + 2 + i for i in range(len(tokens_b))]
				masked_weights = [1] * n_pred
				masked_ids = self.indexer(tokens_b[1:] + ['[SEP]'])
			else:
				# For masked Language Models
				# the number of prediction is sometimes less than max_pred when sequence is short
				effective_length = len(tokens_b)
				if self.mask_source_words:
					effective_length += len(tokens_a)
				n_pred = min(self.max_pred, max(
					1, int(round(effective_length * self.mask_prob))))

				# candidate positions of masked tokens
				cand_pos = []
				special_pos = set()
				for i, tk in enumerate(tokens):
					# only mask tokens_b (target sequence)
					# we will mask [SEP] as an ending symbol
					if (i >= len(tokens_a) + 2) and (tk != '[CLS]'):
						cand_pos.append(i)
					elif self.mask_source_words and (i < len(tokens_a) + 2) and (tk != '[CLS]') and (
					not tk.startswith('[SEP')):
						cand_pos.append(i)
					else:
						special_pos.add(i)
				max_cand_pos = max(cand_pos)

				masked_pos = list([max_cand_pos-1])
				if len(masked_pos) > n_pred:
					shuffle(masked_pos)
					masked_pos = masked_pos[:n_pred]

				masked_tokens = [tokens[pos] for pos in masked_pos]
				for pos in masked_pos:
					if rand() < 0.8:  # 80%
						tokens[pos] = '[MASK]'
					elif rand() < 0.5:  # 10%
						tokens[pos] = get_random_word(self.vocab_words)
				# when n_pred < max_pred, we only calculate loss within n_pred
				masked_weights = [1] * len(masked_tokens)

				# Token Indexing
				masked_ids = self.indexer(masked_tokens)
			# Token Indexing
			input_ids = self.indexer(tokens)

			# Zero Padding
			n_pad = self.max_len - len(input_ids)
			input_ids.extend([0] * n_pad)
			segment_ids.extend([0] * n_pad)

			if self.num_qkv > 1:
				mask_qkv = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
				mask_qkv.extend([0] * n_pad)
			else:
				mask_qkv = None

			input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
			if self.mode == "s2s":
				input_mask[:, :len(tokens_a) + 2].fill_(1)
				second_st, second_end = len(
					tokens_a) + 2, len(tokens_a) + len(tokens_b) + 3
				input_mask[second_st:second_end, second_st:second_end].copy_(
					self._tril_matrix[:second_end - second_st, :second_end - second_st])
			else:
				st, end = 0, len(tokens_a) + len(tokens_b) + 3
				input_mask[st:end, st:end].copy_(self._tril_matrix[:end, :end])

			# Zero Padding for masked target
			if self.max_pred > n_pred:
				n_pad = self.max_pred - n_pred
				if masked_ids is not None:
					masked_ids.extend([0] * n_pad)
				if masked_pos is not None:
					masked_pos.extend([0] * n_pad)
				if masked_weights is not None:
					masked_weights.extend([0] * n_pad)

			tgt_pos = []
			for i, tk in enumerate(tokens):
				if (i >= len(tokens_a) + 2) and (tk != '[CLS]' and tk != '[SEP]'):
					tgt_pos.append(i)

			tgt_pos = tgt_pos[:len(masked_pos)]
			tgt_pad = len(masked_pos) - len(tgt_pos)
			tgt_pos.extend([0] * tgt_pad)

			style_ids = self.indexer(tokens_c)
			style_ids = style_ids[:len(masked_pos)]
			style_pad = len(masked_pos) - len(style_ids)
			style_ids.extend([0] * style_pad)

			input_ids_list.append(input_ids)
			segment_ids_list.append(segment_ids)
			input_mask_list.append(input_mask)
			masked_ids_list.append(masked_ids)
			masked_pos_list.append(masked_pos)
			masked_weights_list.append(masked_weights)
			tgt_pos_list.append(tgt_pos)
			labels_list.append(labels)
			ks_labels_list.append(ks_labels)
			style_ids_list.append(style_ids)
			style_labels_list.append(style_labels)
			check_ids_list.append(check_ids)


		input_mask_list = torch.stack(input_mask_list)
		labels_list = torch.stack(labels_list)
		ks_labels_list = torch.stack(ks_labels_list)
		style_labels_list = torch.tensor(style_labels_list)

		return (input_ids_list, segment_ids_list, input_mask_list, mask_qkv, masked_ids_list, masked_pos_list, masked_weights_list, -1, self.task_idx,
				tgt_pos_list, labels_list, ks_labels_list, style_ids_list, style_labels_list, check_ids_list)


