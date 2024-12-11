import torch
from dataclasses import dataclass
import torch.nn as nn
from torch.nn import functional as F
import math
import os
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("high")


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    batch_size: int = 3


class SelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, config.n_embd*3)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0
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
        
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
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
        self.c_proj.NANOGPT_SCALE_INIT = 1.0
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

        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)


    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2*self.config.n_layer)**-0.5
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=std)
            
    def forward(self, idx, targets=None):
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
        
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else: 
            loss = None
        return logits, loss
  
    def generate(self, x: torch.Tensor, max_len: int = 100) -> torch.Tensor:
        for _ in range(max_len):
            logits = self(x[:, -self.config.block_size:])
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            x = torch.cat([x, next_token], dim=1)
        return x


import tiktoken


class DataLoaderLite:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        data_path = os.path.join(os.path.dirname(__file__), 'data', 'TinyStories-train.txt')

        with open(data_path, 'r') as f:
            data = f.read()
        enc = tiktoken.get_encoding('gpt2')
        tokens = enc.encode(data, allowed_special={"<|endoftext|>"})
        self.tokens = torch.tensor(tokens, dtype=torch.long)

        print(f"Total tokens: {len(self.tokens)}")
        print(f"1 epoch = {len(self.tokens) // (B * T)} batches")

        self.current = 0

    def next_batch(self):
        B, T = self.B, self.T
        buff = self.tokens[self.current:self.current + B * T + 1]
        x = buff[:-1].view(B, T)
        y = buff[1:].view(B, T)
        self.current += B * T

        if self.current + B * T + 1 >= len(self.tokens):
            self.current = 0

        return x, y 
    
def get_lr(it):
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    elif it > max_steps:
        return min_lr
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)
            
if __name__ == "__main__":

    from torch import autocast
    import time
    
    max_steps = 100
    max_lr = 6e-4
    min_lr = 0.1 * max_lr    
    warmup_steps = 10
    
    
    config = GPTConfig(vocab_size=50304)
    
    model = GPT(config).to(device)    
    
    train_loader = DataLoaderLite(config.batch_size, config.block_size)

    model_dict_path = os.path.join(os.path.dirname(__file__), "model.ptl")

    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)\
        
    # Calculate number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    for step in range(max_steps):
        
        

        t0 = time.time()
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        with autocast("cuda", dtype=torch.bfloat16):
            logits, loss = model(x, y)
            
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lr = get_lr(step)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        optimizer.step()
        torch.cuda.synchronize()

        t1 = time.time()
        print(f"Step {step} | Loss: {loss.item():.4f} | lr {lr:.6f} | Norm: {norm:.4f} | Time: {(t1 - t0)*1000:.4f}ms")


    torch.save(model.state_dict(), model_dict_path)
