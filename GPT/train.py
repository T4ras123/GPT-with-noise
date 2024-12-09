import torch
from dataclasses import dataclass
import torch.nn as nn
from torch.nn import functional as F
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class GPTConfig:
	block_size: int = 1024
	vocab_size: int = 50257
	n_layer: int = 12
	n_head: int = 12
	n_embd: int = 768

class SelfAttention(nn.Module):
  def __init__(self, config: GPTConfig):
    super().__init__()
    assert config.n_embd % config.n_head == 0
    self.c_attn = nn.Linear(config.n_embd, config.n_embd*3)
    self.c_proj = nn.Linear(config.n_embd, config.n_embd)
    self.n_head = config.n_head
    self.n_embd = config.n_embd
    self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

  def forward(self, x):
    B, T, C = x.size()
    qkv = self.c_attn(x)
    q, k, v = qkv.split(self.n_embd, dim=2)
    k = k.view(B, T, self.n_head, C //self.n_head).transpose(1, 2)
    q = q.view(B, T, self.n_head, C //self.n_head).transpose(1, 2)
    v = v.view(B, T, self.n_head, C //self.n_head).transpose(1, 2)
    att = (q @ k.transpose(-2, -1)) * (1/math.sqrt(k.size(-1)))
    att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    y = att @ v
    y = y.transpose(1, 2).contiguous().view(B, T, C)


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


class MLP(nn.Module):
	def __init__(self, config: GPTConfig):
		super().__init__()
		self.c_fc = nn.Linear(config.n_embd, config.n_embd*4)
		self.c_proj = nn.Linear(config.n_embd*4, config.n_embd)
		self.gelu = nn.GELU()

	def forward(self, x):
		return self.c_proj(self.act(self.c_fc(x)))


class GPT(nn.Module):
	def __init__(self, config: GPTConfig):
		self.config = config
		self.transformer = nn.ModuleDict(dict(
			wte = nn.Embedding(config.vocab_size, config.n_embd),
			wpe = nn.Embedding(config.block_size, config.n_embd),
			h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
		))
		self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

	def forward(self, x):
	    x = self.transformer['wte'](x) + self.transformer['wpe'](torch.arange(x.size(1), device=x.device))
        for block in self.transformer['h']:
            x = block(x)
        return self.lm_head(x)

    def generate(self, x, max_len=100):
        for _ in range(max_len):
            x = torch.cat([x, self.lm_head(x).argmax(-1)], dim=-1)
        return x
