import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from config import GPT2Config


@torch.jit.script
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415

    NOTE: ripped from nanoGPT
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

@torch.jit.script
def fast_gelu(x):
    """from: https://github.com/hendrycks/GELUs"""
    return 0.5 * x * (1 + torch.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))

@torch.jit.script
def quick_gelu(x):
    """from: https://github.com/hendrycks/GELUs"""
    return torch.sigmoid(1.702 * x) * x

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.emb_size % config.nheads == 0
        
        # stacked W_Q, W_K, W_V
        self.c_attn = nn.Linear(config.emb_size, 3 * config.emb_size, bias=config.bias)
        self.c_proj = nn.Linear(config.emb_size, config.emb_size, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # not sure why Huggingface calls the attention mask 'bias', but I'd like
        # to load their weights. I'm also not sure if adding the extra two dims
        # was to help with broadcasting efficiency or not.
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((config.ctx_size, config.ctx_size), dtype=torch.uint8)).view(
                1, 1, config.ctx_size, config.ctx_size
            ),
        )
        
        self.nheads = config.nheads

    def forward(self, x):
        batch_size, seq_len, emb_size = x.size()
        head_dim = emb_size // self.nheads
        
        Q, K, V = self.c_attn(x).split(emb_size, dim=2)
        
        Q = Q.reshape(batch_size, seq_len, self.nheads, head_dim).transpose(1, 2)
        K = K.reshape(batch_size, seq_len, self.nheads, head_dim).transpose(1, 2)
        V = V.reshape(batch_size, seq_len, self.nheads, head_dim).transpose(1, 2)

        attn = Q @ K.transpose(-2, -1) * (1.0 / math.sqrt(head_dim))
        attn = attn.masked_fill(self.bias[:,:,:seq_len,:seq_len]==0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        attn = self.attn_dropout(attn)

        out = attn @ V
        out = out.transpose(1, 2).reshape(batch_size, seq_len, emb_size)
        
        out = self.c_proj(out)
        out = self.resid_dropout(out)

        return out

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.emb_size, 4 * config.emb_size)
        self.c_proj = nn.Linear(4 * config.emb_size, config.emb_size)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x) # TODO: try faster GeLUs
        x = self.c_proj(x)
        x = self.resid_dropout(x)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.emb_size)
        self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.emb_size)
        self.mlp = MLP(config)

    def forward(self, x):
        x = self.ln_1(x)
        x = x + self.attn(x)
        x = self.ln_2(x)
        x = x + self.mlp(x)

        return x

class GPT2Base(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.emb_size)
        self.wpe = nn.Embedding(config.ctx_size, config.emb_size)
        self.dropout = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.nlayers)])
        self.ln_f = nn.LayerNorm(config.emb_size)

    def forward(self, idx):
        _, seq_len = idx.size()
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device).unsqueeze(0)
        tok_embs = self.wte(idx)
        pos_embs = self.wpe(pos)

        out = self.dropout(tok_embs + pos_embs)
        
        for layer in self.h:
            out = layer(out)
        
        out = self.ln_f(out)

        return out

class GPT2LMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = GPT2Base(config)
        self.lm_head = nn.Linear(config.emb_size, config.vocab_size, bias=False)

        self.apply(self._init_weights) # TODO: figure out how Huggingface does initialization

        for n, p in self.named_parameters():
            if n.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.nlayers))

    def forward(self, x, targets=None):
        x = self.transformer(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            # ignore eot_token in loss calculation (i.e. last element of vocab is '<|endoftext|>')
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_pretrained(cls):
        ''' for GPT-2 117M model only (for now) with a some inspiration from nanoGPT '''
        
        def is_transposed(k):
            ''' helper to determine whether a weight should be transposed '''
            transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
            return any(k.endswith(w) for w in transposed)
        
        config = GPT2Config()
        model = GPT2LMHead(config)
        hfmodel = AutoModelForCausalLM.from_pretrained("gpt2")
        
        sd = model.state_dict()
        hf_sd = hfmodel.state_dict()
        
        sd_keys_hf = hf_sd.keys()
        # we don't need to transfer attention masks
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias') and not k.endswith('.attn.bias')]
        
        for k in sd_keys_hf:
            if is_transposed(k):
                # special treatment for the Conv1D weights we need to transpose
                assert hf_sd[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(hf_sd[k].t())
            else:
                # vanilla copy over the other parameters
                assert hf_sd[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(hf_sd[k])

        return model