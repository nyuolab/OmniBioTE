"""
Heavily adapted from Karpathy's nanoGPT (https://github.com/karpathy/nanoGPT)
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from metrics import compute_tensor_stats
import torch.utils.checkpoint as checkpoint

class SwiGLU(nn.Module):
    """
    Used in LLaMA
    """
    def __init__(self):
        super().__init__()

    def forward(self, h):
        h, gate = h.chunk(2, dim=-1)
        return h * F.silu(gate)

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
    """ RMS Normalization """
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        self.alpha = nn.Parameter(torch.ones(n_embd))
        self.gamma = nn.Parameter(torch.zeros(n_embd))
        self.eps = 1e-5

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        x = self.alpha * (x - mean) / (std + self.eps) + self.gamma
        return x

class SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.autoregressive = config.autoregressive
        self.register_buffer("freqs_cis", precompute_freqs_cis(self.n_embd // self.n_head, config.context_length))

        ### Init weights
        self.c_attn.weight.data.normal_(std=self.config.param_std)
        self.c_proj.weight.data.normal_(std=self.config.param_std)

        if self.config.bias:
            self.c_attn.bias.data.zero_()
            self.c_proj.bias.data.zero_()

    def forward(self, x, attn_mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head) # (B, nh, hs, T)
        q = q.view(B, T, self.n_head, C // self.n_head) # (B, nh, hs, T)
        v = v.view(B, T, self.n_head, C // self.n_head) # (B, nh, hs, T)

        if getattr(self.config, "position_encoding", True) == "rope": # use gettr to maintain compatibility with older versions
            # apply RoPE
            q, k = apply_rotary_emb(q, k, self.freqs_cis)
        
        # transpose (B, nh, T, hs)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, 
                                                            scale=8 / self.n_embd, # Changed for µP
                                                            attn_mask=attn_mask, # We don't attend to padding
                                                            dropout_p=self.dropout if self.training else 0, # training
                                                            is_causal=self.autoregressive)


        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))

        return y

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.swiglu  = SwiGLU()
        self.c_proj  = nn.Linear(4 * config.n_embd // 2, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

        ### Init weights
        self.c_fc.weight.data.normal_(std=config.param_std)
        self.c_proj.weight.data.normal_(std=config.param_std / np.sqrt(2)) # sqrt(2) is because the input dim is double

        if config.bias:
            self.c_fc.bias.data.zero_()
            self.c_proj.bias.data.zero_()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.swiglu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.stats = []

    def forward(self, x, attn_mask, compute_stats=False):
        self.stats = []

        attn = self.attn(self.ln_1(x), attn_mask=attn_mask)
        x = x + attn

        mlp = self.mlp(self.ln_2(x))
        x = x + mlp

        if compute_stats:
            self.stats.append(compute_tensor_stats(attn, "attn.acts"))
            self.stats.append(compute_tensor_stats(mlp, "mlp.acts"))
            self.stats.append(compute_tensor_stats(x, "out"))

            def hook_fn(grad, name):
                self.stats.append(compute_tensor_stats(grad, name))

            if attn.requires_grad:
                attn.register_hook(lambda grad: hook_fn(grad, "attn.grads"))
            if mlp.requires_grad:
                mlp.register_hook(lambda grad: hook_fn(grad, "mlp.grads"))
            if x.requires_grad:
                x.register_hook(lambda grad: hook_fn(grad, "out.grads"))
        
        return x

@dataclass
class OmniBioTAConfig:
    context_length: int = 2048
    vocab_size: int = 2**16
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 1024
    dropout: float = 0.05
    bias: bool = False
    autoregressive: bool = False
    position_encoding: str = "rope"
    memory_efficient: bool = False

class OmniBioTA(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.context_length is not None
        self.config = config
        self.config.lr_scale = 32 / config.n_embd # 32 is arbitrary but makes it easier if you set the overall LR to 0.01 (leads to an LR scale of 3.125e-4 at width 1024)
        self.config.param_std = 1 / np.sqrt(config.n_embd)

        assert config.position_encoding in ["rope", "learned"]

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight.data.normal_(std=1e-5)

        if self.config.position_encoding == "learned":
            self.pos_emb = nn.Parameter(torch.zeros(1, config.context_length, config.n_embd))

        self.stats = []

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
            if self.config.position_encoding == "learned":
                n_params -= self.pos_emb.numel()
        return n_params

    def forward(self, idx, attn_mask=None, return_embeddings=False, compute_stats=False):
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
        assert t <= self.config.context_length, f"Cannot forward sequence of length {t}, block size is only {self.config.context_length}"

        self.stats = []

        def hook_fn(grad, name):
            self.stats.append(compute_tensor_stats(grad, name))

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)

        if compute_stats:
            self.stats.append(compute_tensor_stats(tok_emb, "tok_emb.acts"))

            if tok_emb.requires_grad:
                tok_emb.register_hook(lambda grad: hook_fn(grad, "tok_emb.grad"))

        if self.config.position_encoding == "learned":
            tok_emb = tok_emb + self.pos_emb[:, :t]

        x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            if getattr(self.config, "memory_efficient", False):
                x = checkpoint.checkpoint(block, x, attn_mask, compute_stats, use_reentrant=False)
            else:
                x = block(x, attn_mask=attn_mask, compute_stats=compute_stats)

        emb = self.transformer.ln_f(x)

        if return_embeddings:
            return emb
        else:
            logits = self.lm_head(emb) * self.config.lr_scale

            if compute_stats:
                self.stats.append(compute_tensor_stats(logits, "logits"))
                if logits.requires_grad:
                    logits.register_hook(lambda grad: hook_fn(grad, "logits.grad"))

            return logits
    
    def get_stats(self):
        stats = []
        stats += self.stats
        for i, block in enumerate(self.transformer.h):
            for stat in block.stats:
                stat.name = f"block_{i}.{stat.name}"
                stats.append(stat)

        for name, param in self.named_parameters():
            if param.grad is not None:
                stats.append(compute_tensor_stats(param.grad, name + ".grad"))

        return stats
    
    @staticmethod
    def create_optimizer(model, lr=0.01, weight_decay=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        def fixed_lr(param, name):
            '''
            Determines how LR should scale with a param given muP scaling laws
            '''
            keys = ["lm_head"]
            if any([k in name for k in keys]):
                return True
            if len(param.shape) == 1:
                return True
            
            return False
        
        def no_wd(param, name):
            '''
            Determines which parameters should not have weight decay
            '''
            keys = ["ln", "wte", "pos_emb"]
            if any([k in name for k in keys]):
                return True
            
            return False

        fixed_lr_params = [p for n, p in model.named_parameters() if fixed_lr(p, n) and not no_wd(p, n)]
        inverse_dmodel_params = [p for n, p in model.named_parameters() if not fixed_lr(p, n) and not no_wd(p, n)]
        no_wd_params = [p for n, p in model.named_parameters() if no_wd(p, n)]

        fixed_lr = lr
        inverse_dmodel_lr = lr * 32 / model.config.n_embd
        
        param_groups = [
            {"params": inverse_dmodel_params, "lr": inverse_dmodel_lr,
             "weight_decay": weight_decay * model.config.n_embd / 32}, # scales weight decay by inverse LR to keep weight decay constant (since AdamW weight decay is wd * lr)
            {"params": fixed_lr_params, "lr": fixed_lr, "weight_decay": weight_decay},
            {"params": no_wd_params, "lr": fixed_lr, "weight_decay": 0.0}
        ]

        optimizer = torch.optim.AdamW(param_groups,
                                      betas=(beta1, beta2),
                                      eps=epsilon)
        
        return optimizer

class OmniBindER(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.omnibiota = OmniBioTA(config)
        self.proj_head = nn.Linear(config.n_embd, 1)
        self.stats = np.asarray([0, 1]) # mean and stdev

    def forward(self, idx):
        '''
        Args:
            idx: a torch.LongTensor of shape (b, t) of token indices
        Returns:
            G0: the predicted change in Gibbs free energy for the protein-nucleic acid binding interaction
        '''

        emb = self.omnibiota(idx, return_embeddings=True)[:, 0]
        out = self.proj_head(emb) * (1024 / self.config.n_embd) * self.stats[1] + self.stats[0]

        return out