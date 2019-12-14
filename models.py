# Copyright 2018 Dong-Hyun Lee, Kakao Brain.
# (Strongly inspired by original Google BERT code and Hugging Face's code)
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


""" Transformer Model Classes & Config Class """

import math
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from utils.utils import split_last, merge_last

class Config():
	"Configuration for BERT model"
	vocab_size = None # Size of Vocabulary
	dim = 768 # Dimension of Hidden Layer in Transformer Encoder
	n_layers = 12 # Numher of Hidden Layers
	n_heads = 12 # Numher of Heads in Multi-Headed Attention Layers
	dim_ff = 768*4 # Dimension of Intermediate Layers in Positionwise Feedforward Net
	#activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
	p_drop_hidden = 0.1 # Probability of Dropout of various Hidden Layers
	p_drop_attn = 0.1 # Probability of Dropout of Attention Layers
	max_len = 512 # Maximum Length for Positional Embeddings
	n_segments = 2 # Number of Sentence Segments

	@classmethod
	def from_json(cls, file):
		return cls(**json.load(open(file, "r")))


def gelu(x):
	"Implementation of the gelu activation function by Hugging Face"
	return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
	"A layernorm module in the TF style (epsilon inside the square root)."
	def __init__(self, cfg, variance_epsilon=1e-12):
		super(LayerNorm, self).__init__()
		self.gamma = nn.Parameter(torch.ones(cfg.dim))
		self.beta  = nn.Parameter(torch.zeros(cfg.dim))
		self.variance_epsilon = variance_epsilon

	def forward(self, x):
		u = x.mean(-1, keepdim=True)
		s = (x - u).pow(2).mean(-1, keepdim=True)
		x = (x - u) / torch.sqrt(s + self.variance_epsilon)
		return self.gamma * x + self.beta


class Embeddings(nn.Module):
	"The embedding module from word, position and token_type embeddings."
	def __init__(self, cfg):
		super(Embeddings, self).__init__()
		self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.dim) # token embedding
		self.pos_embed = nn.Embedding(cfg.max_len, cfg.dim) # position embedding
		self.seg_embed = nn.Embedding(cfg.n_segments, cfg.dim) # segment(token type) embedding

		self.norm = LayerNorm(cfg)
		self.drop = nn.Dropout(cfg.p_drop_hidden)

	def forward(self, x, seg):
		seq_len = x.size(1)
		# pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
		# pos = torch.arange(seq_len, device=x.device).float()
		pos = Variable(torch.arange(seq_len).long())
		pos = pos.cuda()
		pos = pos.unsqueeze(0).expand_as(x)
		x = Variable(x)
		seg = Variable(seg)
		te = self.tok_embed(x)
		pe = self.pos_embed(pos)
		se = self.seg_embed(seg)
		e = te + pe + se
		return self.drop(self.norm(e))


class MultiHeadedSelfAttention(nn.Module):
	""" Multi-Headed Dot Product Attention """
	def __init__(self, cfg):
		super(MultiHeadedSelfAttention, self).__init__()
		self.proj_q = nn.Linear(cfg.dim, cfg.dim)
		self.proj_k = nn.Linear(cfg.dim, cfg.dim)
		self.proj_v = nn.Linear(cfg.dim, cfg.dim)
		self.drop = nn.Dropout(cfg.p_drop_attn)
		self.scores = None # for visualization
		self.n_heads = cfg.n_heads

	def forward(self, x, mask):
		q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
		q, k, v = (split_last(x, (self.n_heads, -1)).transpose(1, 2)
				   for x in [q, k, v])
		scores = np.dot(q, k.transpose(-2, -1)) / np.sqrt(k.size(-1))
		if mask is not None:
			mask = mask[:, None, None, :].float()
			scores -= 10000.0 * (1.0 - mask)
		scores = self.drop(F.softmax(scores, dim=-1))
		h = (np.dot(scores, v)).transpose(1, 2).contiguous()
		h = merge_last(h, 2)
		self.scores = scores
		return h


class PositionWiseFeedForward(nn.Module):
	""" FeedForward Neural Networks for each position """
	def __init__(self, cfg):
		super(PositionWiseFeedForward, self).__init__()
		self.fc1 = nn.Linear(cfg.dim, cfg.dim_ff)
		self.fc2 = nn.Linear(cfg.dim_ff, cfg.dim)
		#self.activ = lambda x: activ_fn(cfg.activ_fn, x)

	def forward(self, x):
		# (B, S, D) -> (B, S, D_ff) -> (B, S, D)
		return self.fc2(gelu(self.fc1(x)))


class Block(nn.Module):
	""" Transformer Block """
	def __init__(self, cfg):
		super(Block, self).__init__()
		self.attn = MultiHeadedSelfAttention(cfg)
		self.proj = nn.Linear(cfg.dim, cfg.dim)
		self.norm1 = LayerNorm(cfg)
		self.pwff = PositionWiseFeedForward(cfg)
		self.norm2 = LayerNorm(cfg)
		self.drop = nn.Dropout(cfg.p_drop_hidden)

	def forward(self, x, mask):
		h = self.attn(x, mask)
		h = self.norm1(x + self.drop(self.proj(h)))
		h = self.norm2(h + self.drop(self.pwff(h)))
		return h


class Transformer(nn.Module):
	""" Transformer with Self-Attentive Blocks"""
	def __init__(self, cfg):
		super(Transformer, self).__init__()
		self.embed = Embeddings(cfg)
		self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])   

	def forward(self, x, seg, mask):
		h = self.embed(x, seg)
		for block in self.blocks:
			h = block(h, mask)
		return h


class Classifier(nn.Module):
	""" Classifier with Transformer """
	def __init__(self, cfg, n_labels):
		super(Classifier, self).__init__()
		self.transformer = Transformer(cfg)
		self.fc = nn.Linear(cfg.dim, cfg.dim)
		self.activ = nn.Tanh()
		self.drop = nn.Dropout(cfg.p_drop_hidden)
		self.classifier = nn.Linear(cfg.dim, n_labels)

	def forward(self, input_ids, segment_ids, input_mask):
		h = self.transformer(input_ids, segment_ids, input_mask)
		pooled_h = self.activ(self.fc(h[:, 0]))
		# pooled_h = self.activ(self.fc(Variable(h[:, 0])))
		logits = self.classifier(self.drop(pooled_h))
		# logits = self.classifier(self.drop(Variable(pooled_h)))
		return logits

class Opinion_extract(nn.Module):
	""" Opinion_extraction """
	def __init__(self, cfg, max_len, n_labels):
		super(Opinion_extract, self).__init__()
		self.transformer = Transformer(cfg)
		self.fc = nn.Linear(cfg.dim, cfg.dim)
		self.activ = nn.Tanh()
		self.drop = nn.Dropout(cfg.p_drop_hidden)
		self.extract = nn.Linear(cfg.dim, n_labels)
		self.sigmoid = nn.Sigmoid()

	def forward(self, input_ids, segment_ids, input_mask):
		h = self.transformer(input_ids, segment_ids, input_mask)
		h = self.drop(self.activ(self.fc(h[:, 1:-1])))
		seq_h = self.extract(h)
		seq_h = seq_h.squeeze()
		return self.sigmoid(seq_h)
