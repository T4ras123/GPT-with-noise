import torch
from dataclasses import dataclass
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
	block_size: int = 256
	vocab_size: int = 65
	n_layer: int = 12
	n_head: int = 8
	n_embd: int = 512
	


class Block(nn.Module):
	def __init__(self, config: GPTConfig):
		super().__init__()
		self.ln_1 = nn.LayerNorm(config.n_embd)
		self.ln_2 = nn.LayerNorm(config.n_embd)
		self.attn = SelfAttention(config)
		self.mlp = MLP(config)

	def forward(self, x):
		x = x + self.attn(self.ln_1(x))
		x = x + self.mlp(self.ln_2(x))
		return x

    
class GPT(nn.Module):
	def __init__(self, config: GPTConfig):
		self.config = config
		self.transformer = nn.ModuleDict(dict(
			wte = nn.Embedding(config.vocab_size, config.n_embd),
			wpe = nn.Embedding(config.block_size, config.n_embd),
			h = nn.ModuleList([Block() for _ in range(config.n_layer)]),
			h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
		))
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
		

