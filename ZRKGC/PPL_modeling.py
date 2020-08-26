# coding=utf-8
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import tarfile
import tempfile
import shutil
import numpy as np
from scipy.stats import truncnorm

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.container import ModuleList
from torch.nn.init import xavier_uniform_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from file_utils import cached_path
from loss import LabelSmoothingLoss
import nltk

from tokenization import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("unilm_v2_bert_pretrain", do_lower_case=True)
from nltk.text import TextCollection

logger = logging.getLogger(__name__)

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'




with open("qkr_train.src.10000", encoding="utf-8") as file:
    data = file.readlines()
data = [item.strip().replace(" 。", "") for item in data]
sents = [nltk.word_tokenize(sent) for sent in data]
corpus = TextCollection(sents)

def clean_line_list(line_list):
    new_line_list = []
    for token in line_list:
        if token in ["[CLS]","[SEP]","[PAD]", "[UNK]","[MASK]","<#Q#>","<#Q2K#>"]:
            new_line_list.append(".")
        else:
            new_line_list.append(token)
    return new_line_list

def get_one_idf_score(line_list):
    score = []
    for token in line_list:
        score.append(round(corpus.idf(token),4))
    return score

def get_idf_score(batch_check,batch_r): #list
    batch_check_score = []
    batch_response_score = []
    for b in range(len(batch_check)):
        check_one_list = tokenizer.convert_ids_to_tokens(batch_check[b]) #line to list
        response_one_list = tokenizer.convert_ids_to_tokens(batch_r[b]) #line to list

        check_one_list = clean_line_list(check_one_list)
        response_one_list = clean_line_list(response_one_list)

        batch_check_score.append(get_one_idf_score(check_one_list))
        batch_response_score.append(get_one_idf_score(response_one_list))

    return batch_check_score, batch_response_score

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 relax_projection=0,
                 new_pos_ids=False,
                 initializer_range=0.02,
                 task_idx=None,
                 fp32_embedding=False,
                 ffn_type=0,
                 label_smoothing=None,
                 num_qkv=0,
                 seg_emb=False):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.relax_projection = relax_projection
            self.new_pos_ids = new_pos_ids
            self.initializer_range = initializer_range
            self.task_idx = task_idx
            self.fp32_embedding = fp32_embedding
            self.ffn_type = ffn_type
            self.label_smoothing = label_smoothing
            self.num_qkv = num_qkv
            self.seg_emb = seg_emb
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    print("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex.")

    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-5):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size)
        if hasattr(config, 'fp32_embedding'):
            self.fp32_embedding = config.fp32_embedding
        else:
            self.fp32_embedding = False

        if hasattr(config, 'new_pos_ids') and config.new_pos_ids:
            self.num_pos_emb = 4
        else:
            self.num_pos_emb = 1
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size*self.num_pos_emb)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)


    def get_position_token_type_embedding(self, input_ids, token_type_ids=None, position_ids=None, task_idx=None,relace_embeddings=None,latent_z=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if relace_embeddings == True:
            words_embeddings = self.word_embeddings(input_ids)
            words_embeddings = torch.cat((words_embeddings[:, 0, :].unsqueeze(1), latent_z.type_as(words_embeddings),
                                          words_embeddings[:, 2:, :]), dim=1)
            #print("replace latent_z")
        else:
            words_embeddings = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if self.num_pos_emb > 1:
            num_batch = position_embeddings.size(0)
            num_pos = position_embeddings.size(1)
            position_embeddings = position_embeddings.view(
                num_batch, num_pos, self.num_pos_emb, -1)[torch.arange(0, num_batch).long(), :, task_idx, :]

        embeddings = position_embeddings + token_type_embeddings

        return embeddings


    def get_word_embedding(self, input_ids):

        words_embeddings = self.word_embeddings(input_ids)

        return words_embeddings


    def forward(self, input_ids, token_type_ids=None, position_ids=None, task_idx=None,relace_embeddings=None,latent_z=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if relace_embeddings == True:
            words_embeddings = self.word_embeddings(input_ids)
            words_embeddings = torch.cat((words_embeddings[:,0,:].unsqueeze(1),latent_z.type_as(words_embeddings), words_embeddings[:,2:,:]),dim=1)
            #print("replace latent_z")
        else:
            words_embeddings = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        if self.num_pos_emb > 1:
            num_batch = position_embeddings.size(0)
            num_pos = position_embeddings.size(1)
            position_embeddings = position_embeddings.view(
                num_batch, num_pos, self.num_pos_emb, -1)[torch.arange(0, num_batch).long(), :, task_idx, :]

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        if self.fp32_embedding:
            embeddings = embeddings.half()
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if hasattr(config, 'num_qkv') and (config.num_qkv > 1):
            self.num_qkv = config.num_qkv
        else:
            self.num_qkv = 1

        self.query = nn.Linear(
            config.hidden_size, self.all_head_size*self.num_qkv)
        self.key = nn.Linear(config.hidden_size,
                             self.all_head_size*self.num_qkv)
        self.value = nn.Linear(
            config.hidden_size, self.all_head_size*self.num_qkv)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.uni_debug_flag = True if os.getenv(
            'UNI_DEBUG_FLAG', '') else False
        if self.uni_debug_flag:
            self.register_buffer('debug_attention_probs',
                                 torch.zeros((512, 512)))
        if hasattr(config, 'seg_emb') and config.seg_emb:
            self.b_q_s = nn.Parameter(torch.zeros(
                1, self.num_attention_heads, 1, self.attention_head_size))
            self.seg_emb = nn.Embedding(
                config.type_vocab_size, self.all_head_size)
        else:
            self.b_q_s = None
            self.seg_emb = None

    def transpose_for_scores(self, x, mask_qkv=None):
        if self.num_qkv > 1:
            sz = x.size()[:-1] + (self.num_qkv,
                                  self.num_attention_heads, self.all_head_size)
            # (batch, pos, num_qkv, head, head_hid)
            x = x.view(*sz)
            if mask_qkv is None:
                x = x[:, :, 0, :, :]
            elif isinstance(mask_qkv, int):
                x = x[:, :, mask_qkv, :, :]
            else:
                # mask_qkv: (batch, pos)
                if mask_qkv.size(1) > sz[1]:
                    mask_qkv = mask_qkv[:, :sz[1]]
                # -> x: (batch, pos, head, head_hid)
                x = x.gather(2, mask_qkv.view(sz[0], sz[1], 1, 1, 1).expand(
                    sz[0], sz[1], 1, sz[3], sz[4])).squeeze(2)
        else:
            sz = x.size()[:-1] + (self.num_attention_heads,
                                  self.attention_head_size)
            # (batch, pos, head, head_hid)
            x = x.view(*sz)
        # (batch, head, pos, head_hid)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, history_states=None, mask_qkv=None, seg_ids=None):
        if history_states is None:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)
        else:
            x_states = torch.cat((history_states, hidden_states), dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, mask_qkv)
        key_layer = self.transpose_for_scores(mixed_key_layer, mask_qkv)
        value_layer = self.transpose_for_scores(mixed_value_layer, mask_qkv)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # (batch, head, pos, pos)
        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))

        if self.seg_emb is not None:
            seg_rep = self.seg_emb(seg_ids)
            # (batch, pos, head, head_hid)
            seg_rep = seg_rep.view(seg_rep.size(0), seg_rep.size(
                1), self.num_attention_heads, self.attention_head_size)
            qs = torch.einsum('bnih,bjnh->bnij',
                              query_layer+self.b_q_s, seg_rep)
            attention_scores = attention_scores + qs

        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if self.uni_debug_flag:
            _pos = attention_probs.size(-1)
            self.debug_attention_probs[:_pos, :_pos].copy_(
                attention_probs[0].mean(0).view(_pos, _pos))

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask, history_states=None, mask_qkv=None, seg_ids=None):
        self_output = self.self(
            input_tensor, attention_mask, history_states=history_states, mask_qkv=mask_qkv, seg_ids=seg_ids)
        attention_output = self.output(self_output, input_tensor)
        return attention_output


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class TransformerFFN(nn.Module):
    def __init__(self, config):
        super(TransformerFFN, self).__init__()
        self.ffn_type = config.ffn_type
        assert self.ffn_type in (1, 2)
        if self.ffn_type in (1, 2):
            self.wx0 = nn.Linear(config.hidden_size, config.hidden_size)
        if self.ffn_type in (2,):
            self.wx1 = nn.Linear(config.hidden_size, config.hidden_size)
        if self.ffn_type in (1, 2):
            self.output = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        if self.ffn_type in (1, 2):
            x0 = self.wx0(x)
            if self.ffn_type == 1:
                x1 = x
            elif self.ffn_type == 2:
                x1 = self.wx1(x)
            out = self.output(x0 * x1)
        out = self.dropout(out)
        out = self.LayerNorm(out + x)
        return out


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.ffn_type = config.ffn_type
        if self.ffn_type:
            self.ffn = TransformerFFN(config)
        else:
            self.intermediate = BertIntermediate(config)
            self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, history_states=None, mask_qkv=None, seg_ids=None):
        attention_output = self.attention(
            hidden_states, attention_mask, history_states=history_states, mask_qkv=mask_qkv, seg_ids=seg_ids)
        if self.ffn_type:
            layer_output = self.ffn(attention_output)
        else:
            intermediate_output = self.intermediate(attention_output)
            layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, prev_embedding=None, prev_encoded_layers=None, mask_qkv=None, seg_ids=None):
        # history embedding and encoded layer must be simultanously given
        assert (prev_embedding is None) == (prev_encoded_layers is None)

        all_encoder_layers = []
        if (prev_embedding is not None) and (prev_encoded_layers is not None):
            history_states = prev_embedding
            for i, layer_module in enumerate(self.layer):
                hidden_states = layer_module(
                    hidden_states, attention_mask, history_states=history_states, mask_qkv=mask_qkv, seg_ids=seg_ids)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
                if prev_encoded_layers is not None:
                    history_states = prev_encoded_layers[i]
        else:
            for layer_module in self.layer:
                hidden_states = layer_module(
                    hidden_states, attention_mask, mask_qkv=mask_qkv, seg_ids=seg_ids)
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        hid_size = config.hidden_size
        if hasattr(config, 'relax_projection') and (config.relax_projection > 1):
            hid_size *= config.relax_projection
        self.dense = nn.Linear(config.hidden_size, hid_size)
        self.LayerNorm = BertLayerNorm(hid_size, eps=1e-5)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(
            bert_model_embedding_weights.size(0)))
        if hasattr(config, 'relax_projection') and (config.relax_projection > 1):
            self.relax_projection = config.relax_projection
        else:
            self.relax_projection = 0
        self.fp32_embedding = config.fp32_embedding

        def convert_to_type(tensor):
            if self.fp32_embedding:
                return tensor.half()
            else:
                return tensor
        self.type_converter = convert_to_type
        self.converted = False

    def forward(self, hidden_states, task_idx=None):
        if not self.converted:
            self.converted = True
            if self.fp32_embedding:
                self.transform.half()
        hidden_states = self.transform(self.type_converter(hidden_states))
        if self.relax_projection > 1:
            num_batch = hidden_states.size(0)
            num_pos = hidden_states.size(1)
            # (batch, num_pos, relax_projection*hid) -> (batch, num_pos, relax_projection, hid) -> (batch, num_pos, hid)
            hidden_states = hidden_states.view(
                num_batch, num_pos, self.relax_projection, -1)[torch.arange(0, num_batch).long(), :, task_idx, :]
        if self.fp32_embedding:
            hidden_states = F.linear(self.type_converter(hidden_states), self.type_converter(
                self.decoder.weight), self.type_converter(self.bias))
        else:
            hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights, num_labels=2):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(
            config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, num_labels)

    def forward(self, sequence_output, pooled_output, task_idx=None):
        prediction_scores = self.predictions(sequence_output, task_idx)
        if pooled_output is None:
            seq_relationship_score = None
        else:
            seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class PreTrainedBertModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedBertModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name, state_dict=None, cache_dir=None, *inputs, **kwargs):
        """
        Instantiate a PreTrainedBertModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-base-multilingual`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        if pretrained_model_name in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name]
        else:
            archive_file = pretrained_model_name
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(
                archive_file, cache_dir=cache_dir)
        except FileNotFoundError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file):
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        if ('config_path' in kwargs) and kwargs['config_path']:
            config_file = kwargs['config_path']
        else:
            config_file = os.path.join(serialization_dir, CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)

        # define new type_vocab_size (there might be different numbers of segment ids)
        if 'type_vocab_size' in kwargs:
            config.type_vocab_size = kwargs['type_vocab_size']
        # define new relax_projection
        if ('relax_projection' in kwargs) and kwargs['relax_projection']:
            config.relax_projection = kwargs['relax_projection']
        # new position embedding
        if ('new_pos_ids' in kwargs) and kwargs['new_pos_ids']:
            config.new_pos_ids = kwargs['new_pos_ids']
        # define new relax_projection
        if ('task_idx' in kwargs) and kwargs['task_idx']:
            config.task_idx = kwargs['task_idx']
        # define new max position embedding for length expansion
        if ('max_position_embeddings' in kwargs) and kwargs['max_position_embeddings']:
            config.max_position_embeddings = kwargs['max_position_embeddings']
        # use fp32 for embeddings
        if ('fp32_embedding' in kwargs) and kwargs['fp32_embedding']:
            config.fp32_embedding = kwargs['fp32_embedding']
        # type of FFN in transformer blocks
        if ('ffn_type' in kwargs) and kwargs['ffn_type']:
            config.ffn_type = kwargs['ffn_type']
        # label smoothing
        if ('label_smoothing' in kwargs) and kwargs['label_smoothing']:
            config.label_smoothing = kwargs['label_smoothing']
        # dropout
        if ('hidden_dropout_prob' in kwargs) and kwargs['hidden_dropout_prob']:
            config.hidden_dropout_prob = kwargs['hidden_dropout_prob']
        if ('attention_probs_dropout_prob' in kwargs) and kwargs['attention_probs_dropout_prob']:
            config.attention_probs_dropout_prob = kwargs['attention_probs_dropout_prob']
        # different QKV
        if ('num_qkv' in kwargs) and kwargs['num_qkv']:
            config.num_qkv = kwargs['num_qkv']
        # segment embedding for self-attention
        if ('seg_emb' in kwargs) and kwargs['seg_emb']:
            config.seg_emb = kwargs['seg_emb']
        # initialize word embeddings
        _word_emb_map = None
        if ('word_emb_map' in kwargs) and kwargs['word_emb_map']:
            _word_emb_map = kwargs['word_emb_map']

        logger.info("Model config {}".format(config))

        # clean the arguments in kwargs
        for arg_clean in ('config_path', 'type_vocab_size', 'relax_projection', 'new_pos_ids', 'task_idx', 'max_position_embeddings', 'fp32_embedding', 'ffn_type', 'label_smoothing', 'hidden_dropout_prob', 'attention_probs_dropout_prob', 'num_qkv', 'seg_emb', 'word_emb_map'):
            if arg_clean in kwargs:
                del kwargs[arg_clean]

        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # initialize new segment embeddings
        _k = 'bert.embeddings.token_type_embeddings.weight'
        if (_k in state_dict) and (config.type_vocab_size != state_dict[_k].shape[0]):
            logger.info("config.type_vocab_size != state_dict[bert.embeddings.token_type_embeddings.weight] ({0} != {1})".format(
                config.type_vocab_size, state_dict[_k].shape[0]))
            if config.type_vocab_size > state_dict[_k].shape[0]:
                # state_dict[_k].data = state_dict[_k].data.resize_(config.type_vocab_size, state_dict[_k].shape[1])
                state_dict[_k].resize_(
                    config.type_vocab_size, state_dict[_k].shape[1])
                # L2R
                if config.type_vocab_size >= 3:
                    state_dict[_k].data[2, :].copy_(state_dict[_k].data[0, :])
                # R2L
                if config.type_vocab_size >= 4:
                    state_dict[_k].data[3, :].copy_(state_dict[_k].data[0, :])
                # S2S
                if config.type_vocab_size >= 6:
                    state_dict[_k].data[4, :].copy_(state_dict[_k].data[0, :])
                    state_dict[_k].data[5, :].copy_(state_dict[_k].data[1, :])
                if config.type_vocab_size >= 7:
                    state_dict[_k].data[6, :].copy_(state_dict[_k].data[1, :])
            elif config.type_vocab_size < state_dict[_k].shape[0]:
                state_dict[_k].data = state_dict[_k].data[:config.type_vocab_size, :]

        _k = 'bert.embeddings.position_embeddings.weight'
        n_config_pos_emb = 4 if config.new_pos_ids else 1
        if (_k in state_dict) and (n_config_pos_emb*config.hidden_size != state_dict[_k].shape[1]):
            logger.info("n_config_pos_emb*config.hidden_size != state_dict[bert.embeddings.position_embeddings.weight] ({0}*{1} != {2})".format(
                n_config_pos_emb, config.hidden_size, state_dict[_k].shape[1]))
            assert state_dict[_k].shape[1] % config.hidden_size == 0
            n_state_pos_emb = int(state_dict[_k].shape[1]/config.hidden_size)
            assert (n_state_pos_emb == 1) != (n_config_pos_emb ==
                                              1), "!!!!n_state_pos_emb == 1 xor n_config_pos_emb == 1!!!!"
            if n_state_pos_emb == 1:
                state_dict[_k].data = state_dict[_k].data.unsqueeze(1).repeat(
                    1, n_config_pos_emb, 1).reshape((config.max_position_embeddings, n_config_pos_emb*config.hidden_size))
            elif n_config_pos_emb == 1:
                if hasattr(config, 'task_idx') and (config.task_idx is not None) and (0 <= config.task_idx <= 3):
                    _task_idx = config.task_idx
                else:
                    _task_idx = 0
                state_dict[_k].data = state_dict[_k].data.view(
                    config.max_position_embeddings, n_state_pos_emb, config.hidden_size).select(1, _task_idx)

        # initialize new position embeddings
        _k = 'bert.embeddings.position_embeddings.weight'
        if _k in state_dict and config.max_position_embeddings != state_dict[_k].shape[0]:
            logger.info("config.max_position_embeddings != state_dict[bert.embeddings.position_embeddings.weight] ({0} - {1})".format(
                config.max_position_embeddings, state_dict[_k].shape[0]))
            if config.max_position_embeddings > state_dict[_k].shape[0]:
                old_size = state_dict[_k].shape[0]
                # state_dict[_k].data = state_dict[_k].data.resize_(config.max_position_embeddings, state_dict[_k].shape[1])
                state_dict[_k].resize_(
                    config.max_position_embeddings, state_dict[_k].shape[1])
                start = old_size
                while start < config.max_position_embeddings:
                    chunk_size = min(
                        old_size, config.max_position_embeddings - start)
                    state_dict[_k].data[start:start+chunk_size,
                                        :].copy_(state_dict[_k].data[:chunk_size, :])
                    start += chunk_size
            elif config.max_position_embeddings < state_dict[_k].shape[0]:
                state_dict[_k].data = state_dict[_k].data[:config.max_position_embeddings, :]

        # initialize relax projection
        _k = 'cls.predictions.transform.dense.weight'
        n_config_relax = 1 if (config.relax_projection <
                               1) else config.relax_projection
        if (_k in state_dict) and (n_config_relax*config.hidden_size != state_dict[_k].shape[0]):
            logger.info("n_config_relax*config.hidden_size != state_dict[cls.predictions.transform.dense.weight] ({0}*{1} != {2})".format(
                n_config_relax, config.hidden_size, state_dict[_k].shape[0]))
            assert state_dict[_k].shape[0] % config.hidden_size == 0
            n_state_relax = int(state_dict[_k].shape[0]/config.hidden_size)
            assert (n_state_relax == 1) != (n_config_relax ==
                                            1), "!!!!n_state_relax == 1 xor n_config_relax == 1!!!!"
            if n_state_relax == 1:
                _k = 'cls.predictions.transform.dense.weight'
                state_dict[_k].data = state_dict[_k].data.unsqueeze(0).repeat(
                    n_config_relax, 1, 1).reshape((n_config_relax*config.hidden_size, config.hidden_size))
                for _k in ('cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias'):
                    state_dict[_k].data = state_dict[_k].data.unsqueeze(
                        0).repeat(n_config_relax, 1).view(-1)
            elif n_config_relax == 1:
                if hasattr(config, 'task_idx') and (config.task_idx is not None) and (0 <= config.task_idx <= 3):
                    _task_idx = config.task_idx
                else:
                    _task_idx = 0
                _k = 'cls.predictions.transform.dense.weight'
                state_dict[_k].data = state_dict[_k].data.view(
                    n_state_relax, config.hidden_size, config.hidden_size).select(0, _task_idx)
                for _k in ('cls.predictions.transform.dense.bias', 'cls.predictions.transform.LayerNorm.weight', 'cls.predictions.transform.LayerNorm.bias'):
                    state_dict[_k].data = state_dict[_k].data.view(
                        n_state_relax, config.hidden_size).select(0, _task_idx)

        # initialize QKV
        _all_head_size = config.num_attention_heads * \
            int(config.hidden_size / config.num_attention_heads)
        n_config_num_qkv = 1 if (config.num_qkv < 1) else config.num_qkv
        for qkv_name in ('query', 'key', 'value'):
            _k = 'bert.encoder.layer.0.attention.self.{0}.weight'.format(
                qkv_name)
            if (_k in state_dict) and (n_config_num_qkv*_all_head_size != state_dict[_k].shape[0]):
                logger.info("n_config_num_qkv*_all_head_size != state_dict[_k] ({0}*{1} != {2})".format(
                    n_config_num_qkv, _all_head_size, state_dict[_k].shape[0]))
                for layer_idx in range(config.num_hidden_layers):
                    _k = 'bert.encoder.layer.{0}.attention.self.{1}.weight'.format(
                        layer_idx, qkv_name)
                    assert state_dict[_k].shape[0] % _all_head_size == 0
                    n_state_qkv = int(state_dict[_k].shape[0]/_all_head_size)
                    assert (n_state_qkv == 1) != (n_config_num_qkv ==
                                                  1), "!!!!n_state_qkv == 1 xor n_config_num_qkv == 1!!!!"
                    if n_state_qkv == 1:
                        _k = 'bert.encoder.layer.{0}.attention.self.{1}.weight'.format(
                            layer_idx, qkv_name)
                        state_dict[_k].data = state_dict[_k].data.unsqueeze(0).repeat(
                            n_config_num_qkv, 1, 1).reshape((n_config_num_qkv*_all_head_size, _all_head_size))
                        _k = 'bert.encoder.layer.{0}.attention.self.{1}.bias'.format(
                            layer_idx, qkv_name)
                        state_dict[_k].data = state_dict[_k].data.unsqueeze(
                            0).repeat(n_config_num_qkv, 1).view(-1)
                    elif n_config_num_qkv == 1:
                        if hasattr(config, 'task_idx') and (config.task_idx is not None) and (0 <= config.task_idx <= 3):
                            _task_idx = config.task_idx
                        else:
                            _task_idx = 0
                        assert _task_idx != 3, "[INVALID] _task_idx=3: n_config_num_qkv=1 (should be 2)"
                        if _task_idx == 0:
                            _qkv_idx = 0
                        else:
                            _qkv_idx = 1
                        _k = 'bert.encoder.layer.{0}.attention.self.{1}.weight'.format(
                            layer_idx, qkv_name)
                        state_dict[_k].data = state_dict[_k].data.view(
                            n_state_qkv, _all_head_size, _all_head_size).select(0, _qkv_idx)
                        _k = 'bert.encoder.layer.{0}.attention.self.{1}.bias'.format(
                            layer_idx, qkv_name)
                        state_dict[_k].data = state_dict[_k].data.view(
                            n_state_qkv, _all_head_size).select(0, _qkv_idx)

        if _word_emb_map:
            _k = 'bert.embeddings.word_embeddings.weight'
            for _tgt, _src in _word_emb_map:
                state_dict[_k].data[_tgt, :].copy_(
                    state_dict[_k].data[_src, :])

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        load(model, prefix='' if hasattr(model, 'bert') else 'bert.')
        model.missing_keys = missing_keys
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            logger.info('\n'.join(error_msgs))
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        return model


class BertModel(PreTrainedBertModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLF`) to train on the Next-Sentence task (see BERT's paper).
    ```
    """

    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def rescale_some_parameters(self):
        for layer_id, layer in enumerate(self.encoder.layer):
            layer.attention.output.dense.weight.data.div_(
                math.sqrt(2.0*(layer_id + 1)))
            layer.output.dense.weight.data.div_(math.sqrt(2.0*(layer_id + 1)))

    def get_extended_attention_mask(self, input_ids, token_type_ids, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def get_embedding(self,input_ids,token_type_ids,task_idx=None,relace_embeddings=None,latent_z=None):
        embedding_output = self.embeddings(
            input_ids, token_type_ids, task_idx=task_idx, relace_embeddings=relace_embeddings, latent_z=latent_z)
        return embedding_output

    def get_position_token_type_embedding(self,input_ids,token_type_ids,task_idx=None,relace_embeddings=None,latent_z=None):
        embedding_output = self.embeddings.get_position_token_type_embedding(
            input_ids, token_type_ids, task_idx=task_idx, relace_embeddings=relace_embeddings, latent_z=latent_z)
        return embedding_output

    def get_word_embedding(self,input_ids):
        embedding_output = self.embeddings.get_word_embedding(
            input_ids)
        return embedding_output

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True, mask_qkv=None, task_idx=None,
                relace_embeddings=None, latent_z=None, decode=None, prev_embedding=None, prev_encoded_layers=None, position_ids=None):
        if decode == None:
            extended_attention_mask = self.get_extended_attention_mask(
                input_ids, token_type_ids, attention_mask)

            embedding_output = self.embeddings(
                input_ids, token_type_ids, task_idx=task_idx,relace_embeddings=relace_embeddings, latent_z=latent_z)
            encoded_layers = self.encoder(embedding_output, extended_attention_mask,
                                          output_all_encoded_layers=output_all_encoded_layers, mask_qkv=mask_qkv, seg_ids=token_type_ids)
            sequence_output = encoded_layers[-1]
            pooled_output = self.pooler(sequence_output)
            if not output_all_encoded_layers:
                encoded_layers = encoded_layers[-1]
            return encoded_layers, pooled_output

        else:
            extended_attention_mask = self.get_extended_attention_mask(
                input_ids, token_type_ids, attention_mask)

            embedding_output = self.embeddings(
                input_ids, token_type_ids, position_ids, task_idx=task_idx, relace_embeddings=relace_embeddings,
                latent_z=latent_z)
            encoded_layers = self.encoder(embedding_output,
                                          extended_attention_mask,
                                          output_all_encoded_layers=output_all_encoded_layers,
                                          prev_embedding=prev_embedding,
                                          prev_encoded_layers=prev_encoded_layers, mask_qkv=mask_qkv,
                                          seg_ids=token_type_ids)
            sequence_output = encoded_layers[-1]
            pooled_output = self.pooler(sequence_output)
            if not output_all_encoded_layers:
                encoded_layers = encoded_layers[-1]
            return embedding_output, encoded_layers, pooled_output

class BertModelIncr(BertModel):
    def __init__(self, config):
        super(BertModelIncr, self).__init__(config)

    def get_embedding(self,input_ids,token_type_ids,task_idx=None,relace_embeddings=None,latent_z=None):
        embedding_output = self.embeddings(
            input_ids, token_type_ids, task_idx=task_idx, relace_embeddings=relace_embeddings, latent_z=latent_z)
        return embedding_output

    def forward(self, input_ids, token_type_ids, position_ids, attention_mask, output_all_encoded_layers=True, prev_embedding=None,
                prev_encoded_layers=None, mask_qkv=None, task_idx=None,relace_embeddings=None,latent_z=None):
        extended_attention_mask = self.get_extended_attention_mask(
            input_ids, token_type_ids, attention_mask)

        embedding_output = self.embeddings(
            input_ids, token_type_ids, position_ids, task_idx=task_idx,relace_embeddings=relace_embeddings,latent_z=latent_z)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      prev_embedding=prev_embedding,
                                      prev_encoded_layers=prev_encoded_layers, mask_qkv=mask_qkv, seg_ids=token_type_ids)
        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)
        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return embedding_output, encoded_layers, pooled_output


class BertForPreTraining(PreTrainedBertModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.
    Params:
        config: a BertConfig class instance with the configuration to build a new model.
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.
    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].
    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    model = BertForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertForPreTraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None, mask_qkv=None, task_idx=None):
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False, mask_qkv=mask_qkv, task_idx=task_idx)
        prediction_scores, seq_relationship_score = self.cls(
            sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(
                seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class BertPreTrainingPairTransform(nn.Module):
    def __init__(self, config):
        super(BertPreTrainingPairTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act] \
            if isinstance(config.hidden_act, str) else config.hidden_act
        # self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-5)

    def forward(self, pair_x, pair_y):
        hidden_states = torch.cat([pair_x, pair_y], dim=-1)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        # hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertPreTrainingPairRel(nn.Module):
    def __init__(self, config, num_rel=0):
        super(BertPreTrainingPairRel, self).__init__()
        self.R_xy = BertPreTrainingPairTransform(config)
        self.rel_emb = nn.Embedding(num_rel, config.hidden_size)

    def forward(self, pair_x, pair_y, pair_r, pair_pos_neg_mask):
        # (batch, num_pair, hidden)
        xy = self.R_xy(pair_x, pair_y)
        r = self.rel_emb(pair_r)
        _batch, _num_pair, _hidden = xy.size()
        pair_score = (xy * r).sum(-1)
        # torch.bmm(xy.view(-1, 1, _hidden),r.view(-1, _hidden, 1)).view(_batch, _num_pair)
        # .mul_(-1.0): objective to loss
        return F.logsigmoid(pair_score * pair_pos_neg_mask.type_as(pair_score)).mul_(-1.0)

class TransformerEncoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)


    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

class TransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class BertForPreTrainingLossMask(PreTrainedBertModel):
    """refer to BertForPreTraining"""

    def __init__(self, config, num_labels=2, num_rel=0, num_sentlvl_labels=0, no_nsp=False,
                 mask_word_id = 0,search_beam_size=1, length_penalty=1.0, eos_id=0, sos_id=0,
                forbid_duplicate_ngrams=False, forbid_ignore_set=None, not_predict_set=None, ngram_size=1, min_len=1, mode="s2s", pos_shift=False):
        super(BertForPreTrainingLossMask, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight, num_labels=num_labels)
        self.num_sentlvl_labels = num_sentlvl_labels
        self.cls2 = None
        if self.num_sentlvl_labels > 0:
            self.secondary_pred_proj = nn.Embedding(
                num_sentlvl_labels, config.hidden_size)
            self.cls2 = BertPreTrainingHeads(
                config, self.secondary_pred_proj.weight, num_labels=num_sentlvl_labels)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        if no_nsp:
            self.crit_next_sent = None
        else:
            self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1)
        self.num_labels = num_labels
        self.num_rel = num_rel
        if self.num_rel > 0:
            self.crit_pair_rel = BertPreTrainingPairRel(
                config, num_rel=num_rel)
        if hasattr(config, 'label_smoothing') and config.label_smoothing:
            self.crit_mask_lm_smoothed = LabelSmoothingLoss(
                config.label_smoothing, config.vocab_size, ignore_index=0, reduction='none')
        else:
            self.crit_mask_lm_smoothed = None

        # CVAE parameter
        self.latent_size = config.hidden_size
        self.mu_mlp1 = nn.Linear(2*config.hidden_size, self.latent_size)
        self.var_mlp1 = nn.Linear(2*config.hidden_size, self.latent_size)
        self.mu_mlp2 = nn.Linear(2*config.hidden_size, self.latent_size)
        self.var_mlp2 = nn.Linear(2*config.hidden_size, self.latent_size)
        self.KL_weight = 1

        self.prior_encoder_layer = TransformerEncoderLayer(d_model=768, nhead=12)
        self.prior_transformer_network = TransformerEncoder(self.prior_encoder_layer, num_layers=3)

        self.posterior_encoder_layer = TransformerEncoderLayer(d_model=768, nhead=12)
        self.posterior_transformer_network = TransformerEncoder(self.posterior_encoder_layer, num_layers=3)

        #KS parameter
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        self.mutual_encoder_layer = TransformerEncoderLayer(d_model=768, nhead=12)
        self.mutual_transformer_network = TransformerEncoder(self.mutual_encoder_layer, num_layers=3)
        self.mutual_mlp = nn.Linear(2 * config.hidden_size, 1)

        self.activation = nn.Tanh()
        self.mse_fct = MSELoss()

        self.prob_dense = nn.Linear(2 * config.hidden_size, self.latent_size)

        #TopK to Top1 parameter
        self.Top_K = 1
        self.Topk_encoder_layer = TransformerEncoderLayer(d_model=768, nhead=12)
        self.Topk_transformer_network = TransformerEncoder(self.Topk_encoder_layer, num_layers=3)
        self.Topk_classifier_mlp = nn.Linear(config.hidden_size, 1)

        #Transformer Decoder
        check_encoder_layer = TransformerEncoderLayer(d_model=768, nhead=12)
        self.check_transformer_encoder = TransformerEncoder(check_encoder_layer, num_layers=3)

        check_decoder_layer = TransformerDecoderLayer(d_model=768, nhead=12)
        self.check_transformer_decoder = TransformerDecoder(check_decoder_layer, num_layers=3)
        self.check_mlp = nn.Linear(config.hidden_size,config.vocab_size)

        #Predict Parameter
        predict_transformer_layer = TransformerEncoderLayer(d_model=768, nhead=12)
        self.predict_transformer = TransformerEncoder(predict_transformer_layer, num_layers=3)
        self.predict_mlp = nn.Linear(config.hidden_size,1)

        # Decode parameter
        self.mask_word_id = mask_word_id
        self.search_beam_size = search_beam_size
        self.length_penalty = length_penalty
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.forbid_duplicate_ngrams = forbid_duplicate_ngrams
        self.forbid_ignore_set = forbid_ignore_set
        self.ngram_size = ngram_size
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.pos_shift = pos_shift
        self.not_predict_set = not_predict_set
        self.min_len = min_len


        self.apply(self.init_bert_weights)
        self.bert.rescale_some_parameters()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None, masked_pos=None, masked_weights=None, task_idx=None, pair_x=None,
                pair_x_mask=None, pair_y=None, pair_y_mask=None, pair_r=None, pair_pos_neg_mask=None,
                pair_loss_mask=None, masked_pos_2=None, masked_weights_2=None, masked_labels_2=None,
                num_tokens_a=None, num_tokens_b=None, mask_qkv=None, tgt_pos=None, labels=None,
                ks_labels=None, train_ks=None, train_vae=None, style_ids=None, style_labels=None,
                check_ids=None, pretrain=None, position_ids= None, decode=None):

        #Pretrain Stage
        if pretrain == True:
            if train_ks == None:
                Batch_Size = input_ids.shape[0]
                input_ids = input_ids.reshape(-1, input_ids.shape[2])
                token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[2])
                tgt_pos = tgt_pos.reshape(-1, tgt_pos.shape[2])
                labels = labels.reshape(-1)

                TopK_embedding_output = self.bert.get_embedding(input_ids=input_ids,token_type_ids=token_type_ids)
                TopK_trans_embedding_output = TopK_embedding_output.transpose(0, 1)
                TopK_out = self.Topk_transformer_network(TopK_trans_embedding_output)
                TopK_out = TopK_out[0, :, :]
                TopK_out = self.Topk_classifier_mlp(TopK_out).squeeze().reshape(Batch_Size, -1)
                TopK_prob = torch.softmax(TopK_out, -1)

                choice_know_id = torch.argmax(TopK_prob, 1).detach()

                # recover
                input_ids = input_ids.reshape(Batch_Size, self.Top_K, -1)
                token_type_ids = token_type_ids.reshape(Batch_Size, self.Top_K, -1)
                labels = labels.reshape(Batch_Size, self.Top_K)

                # select ids base choice_know_id
                choice_know_id = choice_know_id.unsqueeze(1)
                input_ids = torch.gather(input_ids, 1,choice_know_id.unsqueeze(2).expand(-1, -1, input_ids.shape[2])).squeeze(1)
                token_type_ids = torch.gather(token_type_ids, 1, choice_know_id.unsqueeze(2).expand(-1, -1,token_type_ids.shape[2])).squeeze(1)

                labels = torch.gather(labels, 1, choice_know_id).squeeze(1)
                check_ids = torch.gather(check_ids, 1,choice_know_id.unsqueeze(2).expand(-1, -1, check_ids.shape[2])).squeeze(1)

            embedding_output = self.bert.get_embedding(input_ids=input_ids, token_type_ids=token_type_ids)
            trans_embedding_output = embedding_output.transpose(0, 1)

            with torch.no_grad():
                check_embedding = self.bert.get_embedding(check_ids, torch.zeros_like(check_ids))
                trans_check_embedding = check_embedding.transpose(0, 1)
                PAD_emb = self.bert.get_word_embedding(torch.tensor([[0] * input_ids.shape[0]]).type_as(input_ids))
                trans_check_embedding = torch.cat((PAD_emb, trans_check_embedding[:-1, :, :]), dim=0)

            trans_embedding_output = torch.cat((PAD_emb, trans_embedding_output[:-1, :, :]), dim=0)
            golden_out = self.mutual_transformer_network(trans_embedding_output)
            golden_out = golden_out[0, :, :]
            golden_out = self.activation(golden_out)

            golden_check_out = self.mutual_transformer_network(trans_check_embedding)
            golden_check_out = golden_check_out[0, :, :]
            golden_check_out = self.activation(golden_check_out)

            golden_out = torch.cat((golden_out, golden_check_out), dim=1)
            golden_out = self.mutual_mlp(golden_out)
            Golden_loss = self.mse_fct(golden_out, labels.unsqueeze(1))

            if pair_x is None or pair_y is None or pair_r is None or pair_pos_neg_mask is None or pair_loss_mask is None:
                return Golden_loss

        if decode == True:
            if self.search_beam_size > 1:
                return self.beam_search(input_ids, token_type_ids, position_ids, attention_mask, task_idx=task_idx, mask_qkv=mask_qkv)

        else:
            # **************** E Step ********************************
            if train_ks == None:
                Batch_Size = input_ids.shape[0]
                input_ids = input_ids.reshape(-1,input_ids.shape[2])
                token_type_ids = token_type_ids.reshape(-1, token_type_ids.shape[2])
                attention_mask = attention_mask.reshape(-1,attention_mask.shape[2],attention_mask.shape[3])
                masked_lm_labels = masked_lm_labels.reshape(-1, masked_lm_labels.shape[2])
                masked_pos = masked_pos.reshape(-1, masked_pos.shape[2])
                masked_weights = masked_weights.reshape(-1, masked_weights.shape[2])
                tgt_pos = tgt_pos.reshape(-1, tgt_pos.shape[2])
                labels = labels.reshape(-1)
                ks_labels = ks_labels.reshape(-1)
                style_ids = style_ids.reshape(-1, style_ids.shape[2])
                style_labels = style_labels.reshape(-1)

                TopK_embedding_output = self.bert.get_embedding(input_ids=input_ids, token_type_ids=token_type_ids)
                TopK_trans_embedding_output = TopK_embedding_output.transpose(0, 1)
                TopK_out = self.Topk_transformer_network(TopK_trans_embedding_output)
                TopK_out = TopK_out[0, :, :]
                TopK_out = self.Topk_classifier_mlp(TopK_out).squeeze().reshape(Batch_Size,-1)
                TopK_prob = torch.softmax(TopK_out,-1)
                choice_know_id = torch.argmax(TopK_prob,1).detach()

                add_embedding = self.bert.get_embedding(input_ids=torch.tensor([[15]]).type_as(input_ids),token_type_ids=torch.tensor([[0]]).type_as(token_type_ids))
                add_embedding = add_embedding.squeeze(1)
                weight_embedding = torch.einsum("i,ij->ij", labels, add_embedding.expand(labels.shape[0], -1).type_as(labels))
                latent_z = weight_embedding.unsqueeze(1)

                sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False,mask_qkv=mask_qkv, task_idx=task_idx, relace_embeddings=True, latent_z=latent_z)

                pooled_output = self.dropout(pooled_output)
                ks_logits = self.classifier(pooled_output)
                ks_logits = F.softmax(ks_logits, dim=-1)
                ks_prob = ks_logits[:, 1]


                latent_z = latent_z.expand(-1, sequence_output.shape[1], -1)
                sequence_output = torch.cat((sequence_output, latent_z), dim=2)
                sequence_output = self.prob_dense(sequence_output)
                sequence_output = self.activation(sequence_output)

                def gather_seq_out_by_pos(seq, pos):
                    return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))

                def batch_loss_mask_and_normalize(loss, mask):
                    mask = mask.type_as(loss)
                    loss = loss * mask
                    denominator = torch.sum(mask, dim=1) + 1e-5
                    return torch.sum(loss, dim=-1) / denominator

                # masked lm
                sequence_output_masked = gather_seq_out_by_pos(
                    sequence_output, masked_pos)
                prediction_scores_masked, seq_relationship_score = self.cls(
                    sequence_output_masked, pooled_output, task_idx=task_idx)
                if self.crit_mask_lm_smoothed:
                    masked_lm_loss = self.crit_mask_lm_smoothed(
                        F.log_softmax(prediction_scores_masked.float(), dim=-1), masked_lm_labels)
                else:
                    masked_lm_loss = self.crit_mask_lm(
                        prediction_scores_masked.transpose(1, 2).float(), masked_lm_labels)
                masked_lm_loss = batch_loss_mask_and_normalize(
                    masked_lm_loss.float(), masked_weights)

                ks_prob = ks_prob.reshape(Batch_Size, self.Top_K).detach()

                lm_prob = F.softmax(-masked_lm_loss.reshape(Batch_Size, self.Top_K), dim=1)
                posterior = (ks_prob * lm_prob / torch.sum(ks_prob * lm_prob, dim=1).unsqueeze(1).expand(-1,lm_prob.shape[1])).detach()

                KL_loss = torch.sum(TopK_prob * (torch.log(TopK_prob) - torch.log(posterior)), dim=1)

                # recover
                input_ids = input_ids.reshape(Batch_Size,self.Top_K, -1)
                token_type_ids = token_type_ids.reshape(Batch_Size,self.Top_K, -1)
                attention_mask = attention_mask.reshape(Batch_Size,self.Top_K, attention_mask.shape[2], -1)
                masked_lm_labels = masked_lm_labels.reshape(Batch_Size,self.Top_K, -1)
                masked_pos = masked_pos.reshape(Batch_Size,self.Top_K, -1)
                masked_weights = masked_weights.reshape(Batch_Size,self.Top_K, -1)
                tgt_pos = tgt_pos.reshape(Batch_Size,self.Top_K, -1)
                labels = labels.reshape(Batch_Size,self.Top_K)
                ks_labels = ks_labels.reshape(Batch_Size,self.Top_K)
                style_ids = style_ids.reshape(Batch_Size,self.Top_K, -1)
                style_labels = style_labels.reshape(Batch_Size,self.Top_K)


                #select ids base choice_know_id
                choice_know_id = choice_know_id.unsqueeze(1)
                input_ids = torch.gather(input_ids, 1, choice_know_id.unsqueeze(2).expand(-1,-1,input_ids.shape[2])).squeeze(1)
                token_type_ids = torch.gather(token_type_ids,1 ,choice_know_id.unsqueeze(2).expand(-1,-1,token_type_ids.shape[2])).squeeze(1)
                attention_mask = torch.gather(attention_mask,1,choice_know_id.unsqueeze(2).unsqueeze(3).expand(-1,-1,attention_mask.shape[2],attention_mask.shape[3])).squeeze(1)
                masked_lm_labels = torch.gather(masked_lm_labels,1,choice_know_id.unsqueeze(2).expand(-1,-1,masked_lm_labels.shape[2])).squeeze(1)
                masked_pos = torch.gather(masked_pos,1,choice_know_id.unsqueeze(2).expand(-1,-1,masked_pos.shape[2])).squeeze(1)
                masked_weights = torch.gather(masked_weights,1,choice_know_id.unsqueeze(2).expand(-1,-1,masked_weights.shape[2])).squeeze(1)
                tgt_pos = torch.gather(tgt_pos,1,choice_know_id.unsqueeze(2).expand(-1,-1,tgt_pos.shape[2])).squeeze(1)
                labels = torch.gather(labels,1,choice_know_id).squeeze(1)
                ks_labels = torch.gather(ks_labels,1,choice_know_id).squeeze(1)
                style_ids = torch.gather(style_ids,1,choice_know_id.unsqueeze(2).expand(-1,-1,style_ids.shape[2])).squeeze(1)
                style_labels = torch.gather(style_labels,1,choice_know_id).squeeze(1)
                check_ids = torch.gather(check_ids, 1, choice_know_id.unsqueeze(2).expand(-1,-1,check_ids.shape[2])).squeeze(1)

            # **************** M Step ********************************
            embedding_output = self.bert.get_embedding(input_ids=input_ids, token_type_ids=token_type_ids)
            trans_embedding_output = embedding_output.transpose(0, 1)

            add_embedding = self.bert.get_embedding(input_ids=torch.tensor([[15]]).type_as(input_ids),token_type_ids=torch.tensor([[0]]).type_as(token_type_ids))
            add_embedding = add_embedding.squeeze(1)
            weight_embedding = torch.einsum("i,ij->ij", labels, add_embedding.expand(labels.shape[0], -1).type_as(labels))

            latent_z = weight_embedding.unsqueeze(1)
            sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False, mask_qkv=mask_qkv, task_idx=task_idx,relace_embeddings=True,latent_z=latent_z)
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)

            #Knowledge Selection Loss
            if ks_labels is not None:
                if ks_labels.dtype == torch.long:
                    loss_fct = CrossEntropyLoss()
                    ks_loss = loss_fct(
                        logits.view(-1, self.num_labels), ks_labels.view(-1))
                elif ks_labels.dtype == torch.half or ks_labels.dtype == torch.float:
                    loss_fct = MSELoss()
                    ks_loss = loss_fct(logits.view(-1), ks_labels.view(-1))
                else:
                    print('unkown ks_labels.dtype')
                    ks_loss = None

                if train_ks is True:
                    return ks_loss,ks_loss
            else:
                return logits


            QK_embedding_output = embedding_output[:, :210, :]
            trans_QK_embedding_output = QK_embedding_output.transpose(0, 1)
            predict_out = self.predict_transformer(trans_QK_embedding_output)
            predict_out = predict_out[0, :, :]
            predict_out = self.activation(predict_out)
            predict_probs = torch.sigmoid(self.predict_mlp(predict_out).squeeze())
            predict_loss = self.mse_fct(predict_probs,labels)


            latent_z = latent_z.expand(-1, sequence_output.shape[1], -1)
            sequence_output = torch.cat((sequence_output, latent_z), dim=2)
            sequence_output = self.prob_dense(sequence_output)
            sequence_output = self.activation(sequence_output)

            def gather_seq_out_by_pos(seq, pos):
                return torch.gather(seq, 1, pos.unsqueeze(2).expand(-1, -1, seq.size(-1)))

            def gather_id_out_by_pos(seq, pos):
                return torch.gather(seq, 1, pos)


            def sample_gumbel(shape, eps=1e-20):
                """Sample from Gumbel(0, 1)"""
                U = torch.rand(shape)
                return -torch.log(-torch.log(U + eps) + eps)

            def gumbel_softmax_sample(logits, temperature):
                """ Draw a sample from the Gumbel-Softmax distribution"""
                y = logits + torch.tensor(sample_gumbel(logits.shape),device = logits.device)
                return F.softmax(y / temperature,dim=-1)

            def gumbel_softmax(logits, temperature, hard=False):
                """Sample from the Gumbel-Softmax distribution and optionally discretize.
                Args:
                  logits: [batch_size, n_class] unnormalized log-probs
                  temperature: non-negative scalar
                  hard: if True, take argmax, but differentiate w.r.t. soft sample y
                Returns:
                  [batch_size, n_class] sample from the Gumbel-Softmax distribution.
                  If hard=True, then the returned sample will be one-hot, otherwise it will
                  be a probabilitiy distribution that sums to 1 across classes
                """
                y = gumbel_softmax_sample(logits, temperature)
                return y

            tgt_sequence_output = gather_seq_out_by_pos(sequence_output, tgt_pos)
            tgt_prob, _ = self.cls(tgt_sequence_output, pooled_output, task_idx=task_idx)
            #tgt_prob_gumbel = gumbel_softmax(tgt_prob, 0.1, hard=False).type_as(self.bert.embeddings.word_embeddings.weight)
            tgt_prob_gumbel = gumbel_softmax(tgt_prob, 200, hard=True).type_as(input_ids).type_as(self.bert.embeddings.word_embeddings.weight)  # B*T*V

            sample_embedding = torch.einsum("ijk,kl->ijl",tgt_prob_gumbel,self.bert.embeddings.word_embeddings.weight)

            with torch.no_grad():
                check_embedding = self.bert.get_embedding(check_ids, torch.zeros_like(check_ids))
                trans_check_embedding = check_embedding.transpose(0, 1)
                PAD_emb = self.bert.get_word_embedding(torch.tensor([[0] * input_ids.shape[0]]).type_as(input_ids))
                trans_check_embedding = torch.cat((PAD_emb, trans_check_embedding[:-1, :, :]), dim=0)

                golden_response_ids = gather_id_out_by_pos(input_ids, tgt_pos)
                golden_response_ids = golden_response_ids * (tgt_pos > 0).type_as(golden_response_ids)

                check_idf_list, response_idf_list = get_idf_score(check_ids.tolist(),golden_response_ids.tolist())
                check_idf_embedding = torch.einsum("ijk,ij->ik",check_embedding, torch.tensor(check_idf_list).to(check_embedding.device).type_as(check_embedding))

            sample_idf_embedding = torch.einsum("ijk,ij->ik",sample_embedding, torch.tensor(response_idf_list).to(sample_embedding.device).type_as(sample_embedding))
            check_idf_embedding = check_idf_embedding / check_ids.shape[1]
            sample_idf_embedding = sample_idf_embedding / golden_response_ids.shape[1]

            cosine_similarity = torch.cosine_similarity(check_idf_embedding, sample_idf_embedding, dim=1)
            cosine_similarity_loss = -(cosine_similarity)

            # Mutual Information Loss
            position_token_type_embedding_output = self.bert.get_position_token_type_embedding(input_ids=input_ids,token_type_ids=token_type_ids)
            response_position_token_type_embedding = gather_seq_out_by_pos(position_token_type_embedding_output,tgt_pos)
            sample_embedding = sample_embedding + response_position_token_type_embedding

            tgt_pos = tgt_pos.unsqueeze(2).expand(-1, -1, sample_embedding.shape[2])
            #new_embedding_output = embedding_output.scatter(1,tgt_pos,sample_embedding)
            new_embedding_output = torch.cat((embedding_output[:, :215, :], sample_embedding, embedding_output[:, -1, :].unsqueeze(1)), dim=1)
            trans_new_embedding_output = new_embedding_output.transpose(0, 1)

            with torch.no_grad():
                trans_new_embedding_output = torch.cat((PAD_emb, trans_new_embedding_output[:-1,:,:]),dim=0)
                mutual_out = self.mutual_transformer_network(trans_new_embedding_output)
                mutual_out = mutual_out[0, :, :]
                mutual_out = self.activation(mutual_out)

                mutual_check_out = self.mutual_transformer_network(trans_check_embedding)
                mutual_check_out = mutual_check_out[0, :, :]
                mutual_check_out = self.activation(mutual_check_out)

                mutual_out = torch.cat((mutual_out,mutual_check_out),dim=1)
                mutual_out = self.mutual_mlp(mutual_out)
                Mutual_loss = self.mse_fct(mutual_out, labels.unsqueeze(1))

            trans_embedding_output = torch.cat((PAD_emb, trans_embedding_output[:-1, :, :]), dim=0)
            golden_out = self.mutual_transformer_network(trans_embedding_output)
            golden_out = golden_out[0,:,:]
            golden_out = self.activation(golden_out)

            golden_check_out = self.mutual_transformer_network(trans_check_embedding)
            golden_check_out = golden_check_out[0, :, :]
            golden_check_out = self.activation(golden_check_out)

            golden_out = torch.cat((golden_out,golden_check_out),dim=1)
            golden_out = self.mutual_mlp(golden_out)
            Golden_loss = self.mse_fct(golden_out, labels.unsqueeze(1))

            def loss_mask_and_normalize(loss, mask):
                mask = mask.type_as(loss)
                loss = loss * mask
                denominator = torch.sum(mask) + 1e-5
                return (loss / denominator).sum()

            if masked_lm_labels is None:
                if masked_pos is None:
                    prediction_scores, seq_relationship_score = self.cls(
                        sequence_output, pooled_output, task_idx=task_idx)
                else:
                    sequence_output_masked = gather_seq_out_by_pos(
                        sequence_output, masked_pos)
                    prediction_scores, seq_relationship_score = self.cls(
                        sequence_output_masked, pooled_output, task_idx=task_idx)
                return prediction_scores, seq_relationship_score


            sequence_output_masked = gather_seq_out_by_pos(sequence_output, masked_pos)
            prediction_scores_masked, seq_relationship_score = self.cls(sequence_output_masked, pooled_output, task_idx=task_idx)
            if self.crit_mask_lm_smoothed:
                masked_lm_loss = self.crit_mask_lm_smoothed(F.log_softmax(prediction_scores_masked.float(), dim=-1), masked_lm_labels)
            else:
                masked_lm_loss = self.crit_mask_lm(prediction_scores_masked.transpose(1, 2).float(), masked_lm_labels)
            masked_lm_loss = loss_mask_and_normalize(masked_lm_loss.float(), masked_weights)

            if self.crit_next_sent is None or next_sentence_label is None:
                next_sentence_loss = 0.0
            else:
                next_sentence_loss = self.crit_next_sent(seq_relationship_score.view(-1, self.num_labels).float(), next_sentence_label.view(-1))


            if pair_x is None or pair_y is None or pair_r is None or pair_pos_neg_mask is None or pair_loss_mask is None:
                return masked_lm_loss, next_sentence_loss, KL_loss, Mutual_loss, Golden_loss, cosine_similarity_loss, predict_loss

    #Early Stop : Decode
    def beam_search(self, input_ids, token_type_ids, position_ids, attention_mask, task_idx=None, mask_qkv=None):
        input_shape = list(input_ids.size())
        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_shape = list(token_type_ids.size())
        output_length = output_shape[1]

        output_ids = []
        prev_embedding = None
        prev_encoded_layers = None
        curr_ids = input_ids
        mask_ids = input_ids.new(batch_size, 1).fill_(self.mask_word_id)
        next_pos = input_length
        if self.pos_shift:
            sos_ids = input_ids.new(batch_size, 1).fill_(self.sos_id)

        K = self.search_beam_size

        total_scores = []
        beam_masks = []
        step_ids = []
        step_back_ptrs = []
        partial_seqs = []
        forbid_word_mask = None
        buf_matrix = None

        source_token_type_ids = token_type_ids[:, :input_length]
        embedding_output = self.bert.get_embedding(input_ids=input_ids, token_type_ids=source_token_type_ids)
        trans_embedding_output = embedding_output.transpose(0, 1)
        prior_out = self.prior_transformer_network(trans_embedding_output)
        prior = prior_out[0, :, :]

        QK_embedding_output = embedding_output[:, :210, :]
        trans_QK_embedding_output = QK_embedding_output.transpose(0, 1)
        predict_out = self.predict_transformer(trans_QK_embedding_output)
        predict_out = predict_out[0, :, :]
        predict_out = self.activation(predict_out)
        predict_probs = torch.sigmoid(self.predict_mlp(predict_out).squeeze())

        bleu = predict_probs


        add_embedding = self.bert.get_embedding(input_ids=torch.tensor([[15]]).type_as(input_ids),
                                                token_type_ids=torch.tensor([[0]]).type_as(token_type_ids))
        add_embedding = add_embedding.squeeze(1)
        add_embedding = torch.einsum("i,ij->ij", bleu, add_embedding.expand(bleu.shape[0], -1).type_as(bleu))

        prior = torch.cat((prior, add_embedding), dim=1)

        prior_mu = self.mu_mlp1(prior)  # B*768
        prior_logvar = self.var_mlp1(prior)

        std = torch.exp(0.5 * prior_logvar)
        eps = torch.randn([prior_logvar.shape[0], self.latent_size], device=prior_mu.device)  # B * hidden
        latent_z = eps * std + prior_mu

        latent_z = add_embedding

        while next_pos < output_length:
            curr_length = list(curr_ids.size())[1]

            if self.pos_shift:
                if next_pos == input_length:
                    x_input_ids = torch.cat((curr_ids, sos_ids), dim=1)
                    start_pos = 0
                else:
                    x_input_ids = curr_ids
                    start_pos = next_pos
            else:
                start_pos = next_pos - curr_length
                x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)

            curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
            curr_attention_mask = attention_mask[:,
                                  start_pos:next_pos + 1, :next_pos + 1]
            curr_position_ids = position_ids[:, start_pos:next_pos + 1]

            if prev_embedding is None:
                latent_z = latent_z.unsqueeze(1)
                beam_latent_z = latent_z
                new_embedding, new_encoded_layers, _ = \
                    self.bert(x_input_ids, curr_token_type_ids, curr_attention_mask,
                              output_all_encoded_layers=True, prev_embedding=prev_embedding,
                              prev_encoded_layers=prev_encoded_layers, mask_qkv=mask_qkv, relace_embeddings=True,
                              latent_z=latent_z,decode=True, position_ids=curr_position_ids)
            else:
                beam_latent_z = latent_z.unsqueeze(1).expand(-1, self.search_beam_size, -1, -1).reshape(-1,latent_z.shape[1],latent_z.shape[2])
                new_embedding, new_encoded_layers, _ = \
                    self.bert(x_input_ids, curr_token_type_ids, curr_attention_mask,
                              output_all_encoded_layers=True, prev_embedding=prev_embedding,
                              prev_encoded_layers=prev_encoded_layers, mask_qkv=mask_qkv,
                              relace_embeddings=False,decode=True,position_ids=curr_position_ids)

            last_hidden = new_encoded_layers[-1][:, -1:, :]

            last_hidden = torch.cat((last_hidden, beam_latent_z), dim=2)
            last_hidden = self.prob_dense(last_hidden)
            last_hidden = self.activation(last_hidden)

            prediction_scores, _ = self.cls(
                last_hidden, None, task_idx=task_idx)
            log_scores = torch.nn.functional.log_softmax(
                prediction_scores, dim=-1)
            if forbid_word_mask is not None:
                log_scores += (forbid_word_mask * -10000.0)
            if self.min_len and (next_pos - input_length + 1 <= self.min_len):
                log_scores[:, :, self.eos_id].fill_(-10000.0)
            if self.not_predict_set:
                for token_id in self.not_predict_set:
                    log_scores[:, :, token_id].fill_(-10000.0)
            kk_scores, kk_ids = torch.topk(log_scores, k=K)
            if len(total_scores) == 0:
                k_ids = torch.reshape(kk_ids, [batch_size, K])
                back_ptrs = torch.zeros(batch_size, K, dtype=torch.long)
                k_scores = torch.reshape(kk_scores, [batch_size, K])
            else:
                last_eos = torch.reshape(
                    beam_masks[-1], [batch_size * K, 1, 1])
                last_seq_scores = torch.reshape(
                    total_scores[-1], [batch_size * K, 1, 1])
                kk_scores += last_eos * (-10000.0) + last_seq_scores
                kk_scores = torch.reshape(kk_scores, [batch_size, K * K])
                k_scores, k_ids = torch.topk(kk_scores, k=K)
                back_ptrs = torch.div(k_ids, K)
                kk_ids = torch.reshape(kk_ids, [batch_size, K * K])
                k_ids = torch.gather(kk_ids, 1, k_ids)
            step_back_ptrs.append(back_ptrs)
            step_ids.append(k_ids)
            beam_masks.append(torch.eq(k_ids, self.eos_id).float())
            total_scores.append(k_scores)

            def first_expand(x):
                input_shape = list(x.size())
                expanded_shape = input_shape[:1] + [1] + input_shape[1:]
                x = torch.reshape(x, expanded_shape)
                repeat_count = [1, K] + [1] * (len(input_shape) - 1)
                x = x.repeat(*repeat_count)
                x = torch.reshape(x, [input_shape[0] * K] + input_shape[1:])
                return x

            def select_beam_items(x, ids):
                id_shape = list(ids.size())
                id_rank = len(id_shape)
                assert len(id_shape) == 2
                x_shape = list(x.size())
                x = torch.reshape(x, [batch_size, K] + x_shape[1:])
                x_rank = len(x_shape) + 1
                assert x_rank >= 2
                if id_rank < x_rank:
                    ids = torch.reshape(
                        ids, id_shape + [1] * (x_rank - id_rank))
                    ids = ids.expand(id_shape + x_shape[1:])
                y = torch.gather(x, 1, ids)
                y = torch.reshape(y, x_shape)
                return y

            is_first = (prev_embedding is None)

            if self.pos_shift:
                if prev_embedding is None:
                    prev_embedding = first_expand(new_embedding)
                else:
                    prev_embedding = torch.cat(
                        (prev_embedding, new_embedding), dim=1)
                    prev_embedding = select_beam_items(
                        prev_embedding, back_ptrs)
                if prev_encoded_layers is None:
                    prev_encoded_layers = [first_expand(
                        x) for x in new_encoded_layers]
                else:
                    prev_encoded_layers = [torch.cat((x[0], x[1]), dim=1) for x in zip(
                        prev_encoded_layers, new_encoded_layers)]
                    prev_encoded_layers = [select_beam_items(
                        x, back_ptrs) for x in prev_encoded_layers]
            else:
                if prev_embedding is None:
                    prev_embedding = first_expand(new_embedding[:, :-1, :])
                else:
                    prev_embedding = torch.cat(
                        (prev_embedding, new_embedding[:, :-1, :]), dim=1)
                    prev_embedding = select_beam_items(
                        prev_embedding, back_ptrs)
                if prev_encoded_layers is None:
                    prev_encoded_layers = [first_expand(
                        x[:, :-1, :]) for x in new_encoded_layers]
                else:
                    prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                           for x in zip(prev_encoded_layers, new_encoded_layers)]
                    prev_encoded_layers = [select_beam_items(
                        x, back_ptrs) for x in prev_encoded_layers]

            curr_ids = torch.reshape(k_ids, [batch_size * K, 1])

            if is_first:
                token_type_ids = first_expand(token_type_ids)
                position_ids = first_expand(position_ids)
                attention_mask = first_expand(attention_mask)
                mask_ids = first_expand(mask_ids)
                if mask_qkv is not None:
                    mask_qkv = first_expand(mask_qkv)

            if self.forbid_duplicate_ngrams:
                wids = step_ids[-1].tolist()
                ptrs = step_back_ptrs[-1].tolist()
                if is_first:
                    partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            partial_seqs.append([wids[b][k]])
                else:
                    new_partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            new_partial_seqs.append(
                                partial_seqs[ptrs[b][k] + b * K] + [wids[b][k]])
                    partial_seqs = new_partial_seqs

                def get_dup_ngram_candidates(seq, n):
                    cands = set()
                    if len(seq) < n:
                        return []
                    tail = seq[-(n - 1):]
                    if self.forbid_ignore_set and any(tk in self.forbid_ignore_set for tk in tail):
                        return []
                    for i in range(len(seq) - (n - 1)):
                        mismatch = False
                        for j in range(n - 1):
                            if tail[j] != seq[i + j]:
                                mismatch = True
                                break
                        if (not mismatch) and not (
                                self.forbid_ignore_set and (seq[i + n - 1] in self.forbid_ignore_set)):
                            cands.add(seq[i + n - 1])
                    return list(sorted(cands))

                if len(partial_seqs[0]) >= self.ngram_size:
                    dup_cands = []
                    for seq in partial_seqs:
                        dup_cands.append(
                            get_dup_ngram_candidates(seq, self.ngram_size))
                    if max(len(x) for x in dup_cands) > 0:
                        if buf_matrix is None:
                            vocab_size = list(log_scores.size())[-1]
                            buf_matrix = np.zeros(
                                (batch_size * K, vocab_size), dtype=float)
                        else:
                            buf_matrix.fill(0)
                        for bk, cands in enumerate(dup_cands):
                            for i, wid in enumerate(cands):
                                buf_matrix[bk, wid] = 1.0
                        forbid_word_mask = torch.tensor(
                            buf_matrix, dtype=log_scores.dtype)
                        forbid_word_mask = torch.reshape(
                            forbid_word_mask, [batch_size * K, 1, vocab_size]).cuda()
                    else:
                        forbid_word_mask = None
            next_pos += 1

        # [(batch, beam)]
        total_scores = [x.tolist() for x in total_scores]
        step_ids = [x.tolist() for x in step_ids]
        step_back_ptrs = [x.tolist() for x in step_back_ptrs]
        # back tracking
        traces = {'pred_seq': [], 'scores': [], 'wids': [], 'ptrs': []}
        for b in range(batch_size):
            # [(beam,)]
            scores = [x[b] for x in total_scores]
            wids_list = [x[b] for x in step_ids]
            ptrs = [x[b] for x in step_back_ptrs]
            traces['scores'].append(scores)
            traces['wids'].append(wids_list)
            traces['ptrs'].append(ptrs)
            # first we need to find the eos frame where all symbols are eos
            # any frames after the eos frame are invalid
            last_frame_id = len(scores) - 1
            for i, wids in enumerate(wids_list):
                if all(wid == self.eos_id for wid in wids):
                    last_frame_id = i
                    break
            max_score = -math.inf
            frame_id = -1
            pos_in_frame = -1

            for fid in range(last_frame_id + 1):
                for i, wid in enumerate(wids_list[fid]):
                    if wid == self.eos_id or fid == last_frame_id:
                        s = scores[fid][i]
                        if self.length_penalty > 0:
                            s /= math.pow((5 + fid + 1) / 6.0,
                                          self.length_penalty)
                        if s > max_score:
                            max_score = s
                            frame_id = fid
                            pos_in_frame = i
            if frame_id == -1:
                traces['pred_seq'].append([0])
            else:
                seq = [wids_list[frame_id][pos_in_frame]]
                for fid in range(frame_id, 0, -1):
                    pos_in_frame = ptrs[fid][pos_in_frame]
                    seq.append(wids_list[fid - 1][pos_in_frame])
                seq.reverse()
                traces['pred_seq'].append(seq)

        def _pad_sequence(sequences, max_len, padding_value=0):
            trailing_dims = sequences[0].size()[1:]
            out_dims = (len(sequences), max_len) + trailing_dims

            out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                out_tensor[i, :length, ...] = tensor
            return out_tensor

        # convert to tensors for DataParallel
        for k in ('pred_seq', 'scores', 'wids', 'ptrs'):
            ts_list = traces[k]
            if not isinstance(ts_list[0], torch.Tensor):
                dt = torch.float if k == 'scores' else torch.long
                ts_list = [torch.tensor(it, dtype=dt) for it in ts_list]
            traces[k] = _pad_sequence(
                ts_list, output_length, padding_value=0).to(input_ids.device)

        return traces


class BertForSeq2SeqDecoder(PreTrainedBertModel):
    """refer to BertForPreTraining"""

    def __init__(self, config, mask_word_id=0, num_labels=2, num_rel=0,
                 search_beam_size=1, length_penalty=1.0, eos_id=0, sos_id=0,
                 forbid_duplicate_ngrams=False, forbid_ignore_set=None, not_predict_set=None, ngram_size=3, min_len=0, mode="s2s", pos_shift=False):
        super(BertForSeq2SeqDecoder, self).__init__(config)
        self.bert = BertModelIncr(config)
        self.cls = BertPreTrainingHeads(
            config, self.bert.embeddings.word_embeddings.weight, num_labels=num_labels)
        self.apply(self.init_bert_weights)
        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')
        self.crit_next_sent = nn.CrossEntropyLoss(ignore_index=-1)
        self.mask_word_id = mask_word_id
        self.num_labels = num_labels
        self.num_rel = num_rel
        if self.num_rel > 0:
            self.crit_pair_rel = BertPreTrainingPairRel(
                config, num_rel=num_rel)
        self.search_beam_size = search_beam_size
        self.length_penalty = length_penalty
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.forbid_duplicate_ngrams = forbid_duplicate_ngrams
        self.forbid_ignore_set = forbid_ignore_set
        self.not_predict_set = not_predict_set
        self.ngram_size = ngram_size
        self.min_len = min_len
        assert mode in ("s2s", "l2r")
        self.mode = mode
        self.pos_shift = pos_shift

        # CVAE parameter  768
        self.latent_size = config.hidden_size
        self.mu_mlp1 = nn.Linear(2 * config.hidden_size, self.latent_size)
        self.var_mlp1 = nn.Linear(2 * config.hidden_size, self.latent_size)
        self.mu_mlp2 = nn.Linear(2 * config.hidden_size, self.latent_size)
        self.var_mlp2 = nn.Linear(2 * config.hidden_size, self.latent_size)
        self.KL_weight = 1

        self.prior_encoder_layer = TransformerEncoderLayer(d_model=768, nhead=12)
        self.prior_transformer_network = TransformerEncoder(self.prior_encoder_layer, num_layers=3)

        ########
        predict_transformer_layer = TransformerEncoderLayer(d_model=768, nhead=12)
        self.predict_transformer = TransformerEncoder(predict_transformer_layer, num_layers=3)
        self.predict_mlp = nn.Linear(config.hidden_size,1)

        self.prob_dense = nn.Linear(2 * config.hidden_size, self.latent_size)
        self.activation = nn.Tanh()

    def forward(self, input_ids, token_type_ids, position_ids, attention_mask, task_idx=None, mask_qkv=None,bleu=None,train_vae=None):
        if self.search_beam_size > 1:
            return self.beam_search(input_ids, token_type_ids, position_ids, attention_mask, task_idx=task_idx, mask_qkv=mask_qkv,bleu=bleu,train_vae=train_vae)

        input_shape = list(input_ids.size())
        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_shape = list(token_type_ids.size())
        output_length = output_shape[1]

        output_ids = []
        prev_embedding = None
        prev_encoded_layers = None
        curr_ids = input_ids
        mask_ids = input_ids.new(batch_size, 1).fill_(self.mask_word_id)
        next_pos = input_length
        if self.pos_shift:
            sos_ids = input_ids.new(batch_size, 1).fill_(self.sos_id)

        source_token_type_ids = token_type_ids[:, :input_length]
        embedding_output = self.bert.get_embedding(input_ids=input_ids, token_type_ids=source_token_type_ids)
        trans_embedding_output = embedding_output.transpose(0, 1)
        prior_out = self.prior_transformer_network(trans_embedding_output)  # S B H
        prior = prior_out[0, :, :]

        bleu = bleu * torch.ones(size=[prior.shape[0]], dtype=torch.float, device=prior.device)  # B

        add_embedding = self.bert.get_embedding(input_ids=torch.tensor([[15]]).type_as(input_ids),
                                                token_type_ids=torch.tensor([[0]]).type_as(token_type_ids))
        add_embedding = add_embedding.squeeze(1)
        add_embedding = torch.einsum("i,ij->ij", bleu, add_embedding.expand(bleu.shape[0], -1).type_as(bleu))  # B H

        prior = torch.cat((prior, add_embedding), dim=1)

        prior_mu = self.mu_mlp1(prior)  # B*768
        prior_logvar = self.var_mlp1(prior)

        std = torch.exp(0.5 * prior_logvar)
        eps = torch.randn([prior_logvar.shape[0], self.latent_size], device=prior_mu.device)  # B * hidden
        latent_z = eps * std + prior_mu

        latent_z = add_embedding
        latent_z = latent_z.unsqueeze(1)

        while next_pos < output_length:
            curr_length = list(curr_ids.size())[1]

            if self.pos_shift:
                if next_pos == input_length:
                    x_input_ids = torch.cat((curr_ids, sos_ids), dim=1)
                    start_pos = 0
                else:
                    x_input_ids = curr_ids
                    start_pos = next_pos
            else:
                start_pos = next_pos - curr_length
                x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)

            curr_token_type_ids = token_type_ids[:, start_pos:next_pos+1]
            curr_attention_mask = attention_mask[:,
                                                 start_pos:next_pos+1, :next_pos+1]
            curr_position_ids = position_ids[:, start_pos:next_pos+1]


            new_embedding, new_encoded_layers, _ = \
                self.bert(x_input_ids, curr_token_type_ids, curr_position_ids, curr_attention_mask,
                          output_all_encoded_layers=True, prev_embedding=prev_embedding,
                          prev_encoded_layers=prev_encoded_layers, mask_qkv=mask_qkv,relace_embeddings=True,latent_z=latent_z)

            last_hidden = new_encoded_layers[-1][:, -1:, :]

            last_hidden = torch.cat((last_hidden, latent_z), dim=2)  # B*1*2H
            last_hidden = self.prob_dense(last_hidden)  # B*1*H
            last_hidden = self.activation(last_hidden)

            prediction_scores, _ = self.cls(
                last_hidden, None, task_idx=task_idx)
            if self.not_predict_set:
                for token_id in self.not_predict_set:
                    prediction_scores[:, :, token_id].fill_(-10000.0)
            _, max_ids = torch.max(prediction_scores, dim=-1)
            output_ids.append(max_ids)

            if self.pos_shift:
                if prev_embedding is None:
                    prev_embedding = new_embedding
                else:
                    prev_embedding = torch.cat(
                        (prev_embedding, new_embedding), dim=1)
                if prev_encoded_layers is None:
                    prev_encoded_layers = [x for x in new_encoded_layers]
                else:
                    prev_encoded_layers = [torch.cat((x[0], x[1]), dim=1) for x in zip(
                        prev_encoded_layers, new_encoded_layers)]
            else:
                if prev_embedding is None:
                    prev_embedding = new_embedding[:, :-1, :]
                else:
                    prev_embedding = torch.cat(
                        (prev_embedding, new_embedding[:, :-1, :]), dim=1)
                if prev_encoded_layers is None:
                    prev_encoded_layers = [x[:, :-1, :]
                                           for x in new_encoded_layers]
                else:
                    prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                           for x in zip(prev_encoded_layers, new_encoded_layers)]
            curr_ids = max_ids
            next_pos += 1

        return torch.cat(output_ids, dim=1)

    def beam_search(self, input_ids, token_type_ids, position_ids, attention_mask, task_idx=None, mask_qkv=None,bleu=None,train_vae=None):
        input_shape = list(input_ids.size())
        batch_size = input_shape[0]
        input_length = input_shape[1]
        output_shape = list(token_type_ids.size())
        output_length = output_shape[1]

        output_ids = []
        prev_embedding = None
        prev_encoded_layers = None
        curr_ids = input_ids
        mask_ids = input_ids.new(batch_size, 1).fill_(self.mask_word_id)
        next_pos = input_length
        if self.pos_shift:
            sos_ids = input_ids.new(batch_size, 1).fill_(self.sos_id)

        K = self.search_beam_size

        total_scores = []
        beam_masks = []
        step_ids = []
        step_back_ptrs = []
        partial_seqs = []
        forbid_word_mask = None
        buf_matrix = None

        source_token_type_ids = token_type_ids[:, :input_length]
        embedding_output = self.bert.get_embedding(input_ids=input_ids, token_type_ids=source_token_type_ids)
        trans_embedding_output = embedding_output.transpose(0, 1)
        prior_out = self.prior_transformer_network(trans_embedding_output)  # S B H
        prior = prior_out[0, :, :]

        #######
        QK_embedding_output = embedding_output[:, :210, :]
        trans_QK_embedding_output = QK_embedding_output.transpose(0, 1)
        predict_out = self.predict_transformer(trans_QK_embedding_output)
        predict_out = predict_out[0, :, :]  # B*H
        predict_out = self.activation(predict_out)
        predict_probs = torch.sigmoid(self.predict_mlp(predict_out).squeeze())  # B

        bleu = predict_probs
        #print(bleu[0:5])
        #########

        #bleu = bleu * torch.ones(size=[prior.shape[0]], dtype=torch.float, device=prior.device)  # B
        add_embedding = self.bert.get_embedding(input_ids=torch.tensor([[15]]).type_as(input_ids),
                                                token_type_ids=torch.tensor([[0]]).type_as(token_type_ids))
        add_embedding = add_embedding.squeeze(1)
        add_embedding = torch.einsum("i,ij->ij", bleu, add_embedding.expand(bleu.shape[0], -1).type_as(bleu)) #B H

        prior = torch.cat((prior, add_embedding), dim=1)

        prior_mu = self.mu_mlp1(prior)  # B*768
        prior_logvar = self.var_mlp1(prior)

        std = torch.exp(0.5 * prior_logvar)
        eps = torch.randn([prior_logvar.shape[0], self.latent_size], device=prior_mu.device)  # B * hidden
        latent_z = eps * std + prior_mu
        #print(latent_z[0, :])

        latent_z = add_embedding

        while next_pos < output_length:
            curr_length = list(curr_ids.size())[1]

            if self.pos_shift:
                if next_pos == input_length:
                    x_input_ids = torch.cat((curr_ids, sos_ids), dim=1)
                    start_pos = 0
                else:
                    x_input_ids = curr_ids
                    start_pos = next_pos
            else:
                start_pos = next_pos - curr_length
                x_input_ids = torch.cat((curr_ids, mask_ids), dim=1)

            curr_token_type_ids = token_type_ids[:, start_pos:next_pos + 1]
            curr_attention_mask = attention_mask[:,
                                                 start_pos:next_pos + 1, :next_pos + 1]
            curr_position_ids = position_ids[:, start_pos:next_pos + 1]

            if prev_embedding is None:
                latent_z = latent_z.unsqueeze(1)
                beam_latent_z = latent_z
                new_embedding, new_encoded_layers, _ = \
                    self.bert(x_input_ids, curr_token_type_ids, curr_position_ids, curr_attention_mask,
                              output_all_encoded_layers=True, prev_embedding=prev_embedding,
                              prev_encoded_layers=prev_encoded_layers, mask_qkv=mask_qkv, relace_embeddings=True,
                              latent_z=latent_z)
            else:
                beam_latent_z = latent_z.unsqueeze(1).expand(-1, self.search_beam_size, -1, -1).reshape(-1, latent_z.shape[1],latent_z.shape[2])
                new_embedding, new_encoded_layers, _ = \
                    self.bert(x_input_ids, curr_token_type_ids, curr_position_ids, curr_attention_mask,
                              output_all_encoded_layers=True, prev_embedding=prev_embedding,
                              prev_encoded_layers=prev_encoded_layers, mask_qkv=mask_qkv, relace_embeddings=False)

            last_hidden = new_encoded_layers[-1][:, -1:, :]

            last_hidden = torch.cat((last_hidden, beam_latent_z), dim=2)  # B*1*2H
            last_hidden = self.prob_dense(last_hidden)  # B*1*H
            last_hidden = self.activation(last_hidden)

            prediction_scores, _ = self.cls(
                last_hidden, None, task_idx=task_idx)
            log_scores = torch.nn.functional.log_softmax(
                prediction_scores, dim=-1)
            if forbid_word_mask is not None:
                log_scores += (forbid_word_mask * -10000.0)
            if self.min_len and (next_pos-input_length+1 <= self.min_len):
                log_scores[:, :, self.eos_id].fill_(-10000.0)
            if self.not_predict_set:
                for token_id in self.not_predict_set:
                    log_scores[:, :, token_id].fill_(-10000.0)
            kk_scores, kk_ids = torch.topk(log_scores, k=K)
            if len(total_scores) == 0:
                k_ids = torch.reshape(kk_ids, [batch_size, K])
                back_ptrs = torch.zeros(batch_size, K, dtype=torch.long)
                k_scores = torch.reshape(kk_scores, [batch_size, K])
            else:
                last_eos = torch.reshape(
                    beam_masks[-1], [batch_size * K, 1, 1])
                last_seq_scores = torch.reshape(
                    total_scores[-1], [batch_size * K, 1, 1])
                kk_scores += last_eos * (-10000.0) + last_seq_scores
                kk_scores = torch.reshape(kk_scores, [batch_size, K * K])
                k_scores, k_ids = torch.topk(kk_scores, k=K)
                back_ptrs = torch.div(k_ids, K)
                kk_ids = torch.reshape(kk_ids, [batch_size, K * K])
                k_ids = torch.gather(kk_ids, 1, k_ids)
            step_back_ptrs.append(back_ptrs)
            step_ids.append(k_ids)
            beam_masks.append(torch.eq(k_ids, self.eos_id).float())
            total_scores.append(k_scores)

            def first_expand(x):
                input_shape = list(x.size())
                expanded_shape = input_shape[:1] + [1] + input_shape[1:]
                x = torch.reshape(x, expanded_shape)
                repeat_count = [1, K] + [1] * (len(input_shape) - 1)
                x = x.repeat(*repeat_count)
                x = torch.reshape(x, [input_shape[0] * K] + input_shape[1:])
                return x

            def select_beam_items(x, ids):
                id_shape = list(ids.size())
                id_rank = len(id_shape)
                assert len(id_shape) == 2
                x_shape = list(x.size())
                x = torch.reshape(x, [batch_size, K] + x_shape[1:])
                x_rank = len(x_shape) + 1
                assert x_rank >= 2
                if id_rank < x_rank:
                    ids = torch.reshape(
                        ids, id_shape + [1] * (x_rank - id_rank))
                    ids = ids.expand(id_shape + x_shape[1:])
                y = torch.gather(x, 1, ids)
                y = torch.reshape(y, x_shape)
                return y

            is_first = (prev_embedding is None)

            if self.pos_shift:
                if prev_embedding is None:
                    prev_embedding = first_expand(new_embedding)
                else:
                    prev_embedding = torch.cat(
                        (prev_embedding, new_embedding), dim=1)
                    prev_embedding = select_beam_items(
                        prev_embedding, back_ptrs)
                if prev_encoded_layers is None:
                    prev_encoded_layers = [first_expand(
                        x) for x in new_encoded_layers]
                else:
                    prev_encoded_layers = [torch.cat((x[0], x[1]), dim=1) for x in zip(
                        prev_encoded_layers, new_encoded_layers)]
                    prev_encoded_layers = [select_beam_items(
                        x, back_ptrs) for x in prev_encoded_layers]
            else:
                if prev_embedding is None:
                    prev_embedding = first_expand(new_embedding[:, :-1, :])
                else:
                    prev_embedding = torch.cat(
                        (prev_embedding, new_embedding[:, :-1, :]), dim=1)
                    prev_embedding = select_beam_items(
                        prev_embedding, back_ptrs)
                if prev_encoded_layers is None:
                    prev_encoded_layers = [first_expand(
                        x[:, :-1, :]) for x in new_encoded_layers]
                else:
                    prev_encoded_layers = [torch.cat((x[0], x[1][:, :-1, :]), dim=1)
                                           for x in zip(prev_encoded_layers, new_encoded_layers)]
                    prev_encoded_layers = [select_beam_items(
                        x, back_ptrs) for x in prev_encoded_layers]

            curr_ids = torch.reshape(k_ids, [batch_size * K, 1])

            if is_first:
                token_type_ids = first_expand(token_type_ids)
                position_ids = first_expand(position_ids)
                attention_mask = first_expand(attention_mask)
                mask_ids = first_expand(mask_ids)
                if mask_qkv is not None:
                    mask_qkv = first_expand(mask_qkv)

            if self.forbid_duplicate_ngrams:
                wids = step_ids[-1].tolist()
                ptrs = step_back_ptrs[-1].tolist()
                if is_first:
                    partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            partial_seqs.append([wids[b][k]])
                else:
                    new_partial_seqs = []
                    for b in range(batch_size):
                        for k in range(K):
                            new_partial_seqs.append(
                                partial_seqs[ptrs[b][k] + b * K] + [wids[b][k]])
                    partial_seqs = new_partial_seqs

                def get_dup_ngram_candidates(seq, n):
                    cands = set()
                    if len(seq) < n:
                        return []
                    tail = seq[-(n-1):]
                    if self.forbid_ignore_set and any(tk in self.forbid_ignore_set for tk in tail):
                        return []
                    for i in range(len(seq) - (n - 1)):
                        mismatch = False
                        for j in range(n - 1):
                            if tail[j] != seq[i + j]:
                                mismatch = True
                                break
                        if (not mismatch) and not(self.forbid_ignore_set and (seq[i + n - 1] in self.forbid_ignore_set)):
                            cands.add(seq[i + n - 1])
                    return list(sorted(cands))

                if len(partial_seqs[0]) >= self.ngram_size:
                    dup_cands = []
                    for seq in partial_seqs:
                        dup_cands.append(
                            get_dup_ngram_candidates(seq, self.ngram_size))
                    if max(len(x) for x in dup_cands) > 0:
                        if buf_matrix is None:
                            vocab_size = list(log_scores.size())[-1]
                            buf_matrix = np.zeros(
                                (batch_size * K, vocab_size), dtype=float)
                        else:
                            buf_matrix.fill(0)
                        for bk, cands in enumerate(dup_cands):
                            for i, wid in enumerate(cands):
                                buf_matrix[bk, wid] = 1.0
                        forbid_word_mask = torch.tensor(
                            buf_matrix, dtype=log_scores.dtype)
                        forbid_word_mask = torch.reshape(
                            forbid_word_mask, [batch_size * K, 1, vocab_size]).cuda()
                    else:
                        forbid_word_mask = None
            next_pos += 1

        # [(batch, beam)]
        total_scores = [x.tolist() for x in total_scores]
        step_ids = [x.tolist() for x in step_ids]
        step_back_ptrs = [x.tolist() for x in step_back_ptrs]
        # back tracking
        traces = {'pred_seq': [], 'scores': [], 'wids': [], 'ptrs': []}
        for b in range(batch_size):
            # [(beam,)]
            scores = [x[b] for x in total_scores]
            wids_list = [x[b] for x in step_ids]
            ptrs = [x[b] for x in step_back_ptrs]
            traces['scores'].append(scores)
            traces['wids'].append(wids_list)
            traces['ptrs'].append(ptrs)
            # first we need to find the eos frame where all symbols are eos
            # any frames after the eos frame are invalid
            last_frame_id = len(scores) - 1
            for i, wids in enumerate(wids_list):
                if all(wid == self.eos_id for wid in wids):
                    last_frame_id = i
                    break
            max_score = -math.inf
            frame_id = -1
            pos_in_frame = -1

            for fid in range(last_frame_id + 1):
                for i, wid in enumerate(wids_list[fid]):
                    if wid == self.eos_id or fid == last_frame_id:
                        s = scores[fid][i]
                        if self.length_penalty > 0:
                            s /= math.pow((5 + fid + 1) / 6.0,
                                          self.length_penalty)
                        if s > max_score:
                            max_score = s
                            frame_id = fid
                            pos_in_frame = i
            if frame_id == -1:
                traces['pred_seq'].append([0])
            else:
                seq = [wids_list[frame_id][pos_in_frame]]
                for fid in range(frame_id, 0, -1):
                    pos_in_frame = ptrs[fid][pos_in_frame]
                    seq.append(wids_list[fid - 1][pos_in_frame])
                seq.reverse()
                traces['pred_seq'].append(seq)

        def _pad_sequence(sequences, max_len, padding_value=0):
            trailing_dims = sequences[0].size()[1:]
            out_dims = (len(sequences), max_len) + trailing_dims

            out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                out_tensor[i, :length, ...] = tensor
            return out_tensor

        # convert to tensors for DataParallel
        for k in ('pred_seq', 'scores', 'wids', 'ptrs'):
            ts_list = traces[k]
            if not isinstance(ts_list[0], torch.Tensor):
                dt = torch.float if k == 'scores' else torch.long
                ts_list = [torch.tensor(it, dtype=dt) for it in ts_list]
            traces[k] = _pad_sequence(
                ts_list, output_length, padding_value=0).to(input_ids.device)

        return traces

