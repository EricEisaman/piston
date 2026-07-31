"""Reference PyTorch modules matching exportable Browser Train transformers."""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn


ActivationName = Literal["gelu", "relu", "relu2", "silu", "sigmoid", "tanh"]


def _act(name: str):
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "relu2":
        return lambda x: torch.square(torch.relu(x))
    if name == "silu":
        return nn.SiLU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "tanh":
        return nn.Tanh()
    return nn.GELU()


class MLP(nn.Module):
    def __init__(self, n_embed: int, expansion: float, gated: bool, activation: str):
        super().__init__()
        hidden = int(expansion * n_embed)
        self.gated = gated
        self.upProj = nn.Linear(n_embed, hidden)
        self.downProj = nn.Linear(hidden, n_embed)
        self.gateProj = nn.Linear(n_embed, hidden) if gated else None
        self.act = _act(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.upProj(x))
        if self.gateProj is not None:
            h = h * self.gateProj(x)
        return self.downProj(h)


class SelfAttention(nn.Module):
    def __init__(self, n_embed: int, n_heads: int, causal: bool):
        super().__init__()
        assert n_embed % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.causal = causal
        self.cAttn = nn.Linear(n_embed, 3 * n_embed)
        self.cProj = nn.Linear(n_embed, n_embed)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.cAttn(x)
        q, k, v = qkv.split(c, dim=-1)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if self.causal:
            mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
            att = att.masked_fill(mask, float("-inf"))
        if attn_mask is not None:
            # attn_mask: [B, T] with 1=keep — broadcast over heads/queries
            att = att.masked_fill(attn_mask[:, None, None, :] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(b, t, c)
        return self.cProj(y)


class CrossAttention(nn.Module):
    def __init__(self, n_embed: int, n_heads: int):
        super().__init__()
        assert n_embed % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = n_embed // n_heads
        self.qProj = nn.Linear(n_embed, n_embed)
        self.kvProj = nn.Linear(n_embed, 2 * n_embed)
        self.cProj = nn.Linear(n_embed, n_embed)

    def forward(
        self, x: torch.Tensor, memory: torch.Tensor, memory_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, t, c = x.shape
        s = memory.size(1)
        q = self.qProj(x).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.kvProj(memory)
        k, v = kv.split(c, dim=-1)
        k = k.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        if memory_mask is not None:
            att = att.masked_fill(memory_mask[:, None, None, :] == 0, float("-inf"))
        att = torch.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(b, t, c)
        return self.cProj(y)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        n_embed: int,
        n_heads: int,
        expansion: float,
        gated: bool,
        activation: str,
        cross_attention: bool,
        norm_eps: float = 1e-5,
        causal: bool = True,
        attn_name: str = "selfAttn",
    ):
        super().__init__()
        # Encoder MLM layers use `attn` / `lnAttn` names to match Browser Train state dicts.
        self._use_attn_alias = attn_name == "attn"
        if self._use_attn_alias:
            self.lnAttn = nn.LayerNorm(n_embed, eps=norm_eps)
            self.attn = SelfAttention(n_embed, n_heads, causal=causal)
        else:
            self.lnSelfAttn = nn.LayerNorm(n_embed, eps=norm_eps)
            self.selfAttn = SelfAttention(n_embed, n_heads, causal=causal)
        self.crossAttn = CrossAttention(n_embed, n_heads) if cross_attention else None
        self.lnCrossAttn = nn.LayerNorm(n_embed, eps=norm_eps) if cross_attention else None
        self.lnMlp = nn.LayerNorm(n_embed, eps=norm_eps)
        self.mlp = MLP(n_embed, expansion, gated, activation)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._use_attn_alias:
            x = x + self.attn(self.lnAttn(x), attn_mask=attn_mask)
        else:
            x = x + self.selfAttn(self.lnSelfAttn(x), attn_mask=attn_mask)
        if self.crossAttn is not None and memory is not None and self.lnCrossAttn is not None:
            x = x + self.crossAttn(self.lnCrossAttn(x), memory, memory_mask)
        x = x + self.mlp(self.lnMlp(x))
        return x


class DecoderLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        n_heads: int,
        n_layers: int,
        block_size: int,
        expansion: float,
        gated: bool,
        activation: str,
        use_pos_emb: bool,
    ):
        super().__init__()
        self.block_size = block_size
        self.wordEmbedding = nn.Embedding(vocab_size, n_embed)
        self.positionEmbedding = nn.Embedding(block_size, n_embed) if use_pos_emb else None
        self.layers = nn.ModuleList(
            [
                DecoderLayer(n_embed, n_heads, expansion, gated, activation, cross_attention=False)
                for _ in range(n_layers)
            ]
        )
        self.lnF = nn.LayerNorm(n_embed)
        self.lmHead = nn.Linear(n_embed, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _b, t = input_ids.shape
        x = self.wordEmbedding(input_ids)
        if self.positionEmbedding is not None:
            pos = torch.arange(t, device=input_ids.device)
            x = x + self.positionEmbedding(pos)[None, :, :]
        for layer in self.layers:
            x = layer(x)
        x = self.lnF(x)
        return self.lmHead(x)


class EncoderDecoderLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        n_heads: int,
        n_encoder: int,
        n_decoder: int,
        block_src: int,
        block_tgt: int,
        expansion: float,
        gated: bool,
        activation: str,
        use_pos_emb: bool,
    ):
        super().__init__()
        self.block_src = block_src
        self.block_tgt = block_tgt
        self.encWordEmbedding = nn.Embedding(vocab_size, n_embed)
        self.decWordEmbedding = nn.Embedding(vocab_size, n_embed)
        self.encPos = nn.Embedding(block_src, n_embed) if use_pos_emb else None
        self.decPos = nn.Embedding(block_tgt, n_embed) if use_pos_emb else None
        self.encoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    n_embed,
                    n_heads,
                    expansion,
                    gated,
                    activation,
                    cross_attention=False,
                    causal=False,
                )
                for _ in range(n_encoder)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(n_embed, n_heads, expansion, gated, activation, cross_attention=True)
                for _ in range(n_decoder)
            ]
        )
        self.encLn = nn.LayerNorm(n_embed)
        self.decLn = nn.LayerNorm(n_embed)
        self.lmHead = nn.Linear(n_embed, vocab_size, bias=False)

    def forward(
        self, encoder_input_ids: torch.Tensor, decoder_input_ids: torch.Tensor
    ) -> torch.Tensor:
        tt = decoder_input_ids.size(1)
        h = self.encode(encoder_input_ids)
        x = self.decWordEmbedding(decoder_input_ids)
        if self.decPos is not None:
            x = x + self.decPos(torch.arange(tt, device=x.device))[None, :, :]
        for layer in self.decoder_layers:
            x = layer(x, memory=h)
        x = self.decLn(x)
        return self.lmHead(x)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _b, ts = input_ids.shape
        h = self.encWordEmbedding(input_ids)
        if self.encPos is not None:
            h = h + self.encPos(torch.arange(ts, device=h.device))[None, :, :]
        for layer in self.encoder_layers:
            # Bidirectional encoder block (no cross-attn); keep attention_mask in the ONNX graph.
            h = layer(h, attn_mask=attention_mask)
        return self.encLn(h)

    def decode(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tt = input_ids.size(1)
        x = self.decWordEmbedding(input_ids)
        if self.decPos is not None:
            x = x + self.decPos(torch.arange(tt, device=x.device))[None, :, :]
        for layer in self.decoder_layers:
            x = layer(x, memory=encoder_hidden_states, memory_mask=encoder_attention_mask)
        x = self.decLn(x)
        return self.lmHead(x)


class EncDecEncoderExport(nn.Module):
    """BART-style encoder graph: input_ids + attention_mask → last_hidden_state."""

    def __init__(self, inner: EncoderDecoderLM):
        super().__init__()
        self.inner = inner

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.inner.encode(input_ids, attention_mask)


class EncDecDecoderExport(nn.Module):
    """BART-style decoder graph (no past): decoder ids + encoder states → logits."""

    def __init__(self, inner: EncoderDecoderLM):
        super().__init__()
        self.inner = inner

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.inner.decode(input_ids, encoder_hidden_states, encoder_attention_mask)


class EncoderLM(nn.Module):
    """Bidirectional encoder + MLM head for Dyck / encoder-only export."""

    def __init__(
        self,
        vocab_size: int,
        n_embed: int,
        n_heads: int,
        n_layers: int,
        block_size: int,
        expansion: float,
        gated: bool,
        activation: str,
        use_pos_emb: bool,
        type_vocab_size: int = 2,
    ):
        super().__init__()
        self.block_size = block_size
        self.wordEmbedding = nn.Embedding(vocab_size, n_embed)
        self.positionEmbedding = nn.Embedding(block_size, n_embed) if use_pos_emb else None
        self.tokenTypeEmbedding = nn.Embedding(type_vocab_size, n_embed)
        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    n_embed,
                    n_heads,
                    expansion,
                    gated,
                    activation,
                    cross_attention=False,
                    causal=False,
                    attn_name="attn",
                )
                for _ in range(n_layers)
            ]
        )
        self.layerNorm = nn.LayerNorm(n_embed)
        self.transform = nn.Linear(n_embed, n_embed)
        self.mlmLn = nn.LayerNorm(n_embed)
        self.lmHead = nn.Linear(n_embed, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _b, t = input_ids.shape
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        x = self.wordEmbedding(input_ids) + self.tokenTypeEmbedding(token_type_ids)
        if self.positionEmbedding is not None:
            x = x + self.positionEmbedding(torch.arange(t, device=input_ids.device))[None, :, :]
        for layer in self.layers:
            x = layer(x, attn_mask=attention_mask)
        x = self.layerNorm(x)
        h = torch.nn.functional.gelu(self.transform(x))
        h = self.mlmLn(h)
        return self.lmHead(h)
