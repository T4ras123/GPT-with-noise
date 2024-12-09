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
        return self.c_proj(y)

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
        return self.c_proj(self.gelu(self.c_fc(x)))


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()  
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size 
        pos = torch.arange(T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer['wpe'](pos)
        tok_emb = self.transformer['wte'](idx)
        x = pos_emb + tok_emb
        for block in self.transformer['h']:
            x = block(x)
        x = self.transformer['ln_f'](x)
        logits = self.lm_head(x)
        return logits
  
    def generate(self, x: torch.Tensor, max_len: int = 100) -> torch.Tensor:
        for _ in range(max_len):
            logits = self(x[:, -self.config.block_size:])
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            x = torch.cat([x, next_token], dim=1)
        return x


num_return_sequences = 5
max_len = 100
config = GPTConfig()
model = GPT(config).to(device)

import tiktoken
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode('Hello, i\'m a language model,')
tokens = torch.tensor(tokens, dtype=torch.long, device=device)
tokens = torch.unsqueeze(tokens, 0).repeat(num_return_sequences, 1)
x = tokens.to(device)


while x.size(1) < max_len:
    with torch.no_grad():
        logits = model(x)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 5, dim=-1)
        ix = torch.multinomial(topk_probs, 1)
        xcol = topk_indices.gather(1, ix)
        x = toch.cat([x, xcol], dim=1)
        
        
for i in range(num_return_sequences):
    print(f"Sample {i+1}: {enc.decode(x[i].tolist())}")