"""
Heavily adapted from Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT)

Main changes:
    - Removed causal attention
    - Changed SoftMax scaling to match µP
    - Forced FlashAttention
    - Added attention mask to prevent padding from being attended to
    - Removed functions extraneous to this work
    - Disabled weight tying
    - Added RoPE
"""
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from mup import MuReadout

from torch.utils.checkpoint import checkpoint

@torch.jit.script
def fused_gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))

# Modified from facebookresearch/llama/model.py
def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    #assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    assert freqs_cis.shape[-1] == x.shape[-1] # we allow variable sequence lengths
    freqs_cis = freqs_cis[:x.shape[1]] # truncate to sequence length
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


# Taken from facebookresearch/llama/model.py
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

# Taken from facebookresearch/llama/model.py
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (
        theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    return freqs_cis

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout, inplace=True)
        self.resid_dropout = nn.Dropout(config.dropout, inplace=True)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.autoregressive = config.autoregressive
        self.flash = config.flash
        self.register_buffer("freqs_cis", precompute_freqs_cis(self.n_embd // self.n_head, config.block_size))
        
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, attn_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head) # (B, nh, hs, T)
        q = q.view(B, T, self.n_head, C // self.n_head) # (B, nh, hs, T)
        v = v.view(B, T, self.n_head, C // self.n_head) # (B, nh, hs, T)

        # apply RoPE
        q, k = apply_rotary_emb(q, k, self.freqs_cis)
        
        # transpose
        k = k.transpose(1, 2) # (B, nh, T, hs)
        q = q.transpose(1, 2) # (B, nh, T, hs)
        v = v.transpose(1, 2) # (B, nh, T, hs)

        if attn_mask is None:
            # efficient attention using Flash Attention CUDA kernels
            if self.flash:
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
                                                                    scale=8 / self.n_embd, # Changed for µP
                                                                    attn_mask=attn_mask, # We don't attend to padding
                                                                    dropout_p=self.dropout if self.training else 0, # training
                                                                    is_causal=self.autoregressive)
            else:
                # manual implementation of attention
                att = (q @ k.transpose(-2, -1)) * (8 / self.n_embd)
                if self.autoregressive:
                    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        else:
            if self.flash:
                # efficient attention using Flash Attention CUDA kernels
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
                                                                    scale=8 / self.n_embd, # Changed for µP
                                                                    attn_mask=attn_mask, # We don't attend to padding
                                                                    dropout_p=self.dropout if self.training else 0, # training
                                                                    is_causal=False) # if there's an attention mask, this needs to be set to False
            else:
                # manual implementation of attention
                att = (q @ k.transpose(-2, -1)) * (8 / self.n_embd)
                att += attn_mask # add the attention mask
                att = F.softmax(att, dim=-1)
                att = self.attn_dropout(att)
                y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)


        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        #self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout, inplace=True)

    def forward(self, x):
        x = self.c_fc(x)
        #x = self.gelu(x)
        x = fused_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class OmniBioTAConfig:
    block_size: int = 2048
    vocab_size: int = 2**16
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 1024
    dropout: float = 0.1
    bias: bool = False
    autoregressive: bool = False
    checkpoint_freq: int = 0

class OmniBioTA(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout, inplace=True),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = MuReadout(config.n_embd, config.vocab_size, bias=False) # replaced nn.Linear with MuReadout

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wte.weight.numel()
        return n_params

    def forward(self, idx, attn_mask=None, return_embeddings=False):
        '''
        Args:
            idx: a torch.LongTensor of shape (b, t) of token indices
            attn_mask: the attention mask to apply during flash attention
            return_embeddings: if True, return the token embeddings instead of the logits
        Returns:
        if return_embeddings=False:
            logits: a torch.FloatTensor of shape (b, t, vocab_size) of logits
        if return_embeddings=True:
            emb: a torch.FloatTensor of shape (b, t, n_embd) of token embeddings
        '''
        _, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = self.transformer.drop(tok_emb)
        for i, block in enumerate(self.transformer.h):
            if self.config.checkpoint_freq > 0 and i % self.config.checkpoint_freq == 0:
                x = checkpoint(block, x, attn_mask, use_reentrant=False)
            else:
                x = block(x, attn_mask=attn_mask)
        emb = self.transformer.ln_f(x)

        if return_embeddings:
            return emb
        else:
            logits = self.lm_head(emb)
            return logits
    
    def encode(self, idx, method="mean"):
        """
        Encode a sequence of tokens into a single vector representation.
        
        Args:
            idx: a torch.LongTensor of shape (b, t) of token indices
            method: one of "mean", "first", "last", "max", "all"
        Returns:
            a torch.FloatTensor of shape (b, n_embd) of the encoded sequence
        """
        assert method in ["mean", "first", "last", "max", "all"], f"Unknown pooling method {method}"

        emb = self.forward(idx, return_embeddings=True)
        if method == "mean":
            return emb.mean(dim=1)
        elif method == "first":
            return emb[:, 0]
        elif method == "last":
            return emb[:, -1]
        elif method == "max":
            return emb.max(dim=1)[0]
        elif method == "all":
            return emb