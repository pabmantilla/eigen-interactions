"""Monkey-patches for AlphaGenome PyTorch to make all nonlinear ops hookable
by tangermeme's DeepLIFT/SHAP.

AlphaGenome uses functional activations (F.gelu, F.relu, torch.tanh, F.softmax)
which tangermeme can't hook. This module replaces them with nn.Module equivalents.

Usage:
    from ag_deeplift_patches import patch_alphagenome, AGCustomGELU
    patch_alphagenome()  # call once before loading any model
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import alphagenome_pytorch.layers as ag_layers
import alphagenome_pytorch.attention as ag_attn
from alphagenome_pytorch.convolutions import ConvBlock
from alphagenome_pytorch.model import SequenceEncoder
from alphagenome_pytorch.attention import (
    apply_rope, _central_mask_features, _shift, _MAX_RELATIVE_DISTANCE,
)

_PATCHED = False


class AGCustomGELU(nn.Module):
    """Module version of alphagenome_pytorch.layers.gelu (sigmoid approx)."""
    def forward(self, x):
        return torch.sigmoid(1.702 * x) * x


def patch_alphagenome():
    """Apply all monkey-patches. Safe to call multiple times (idempotent)."""
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    # ------------------------------------------------------------------
    # Pool1d: F.max_pool1d -> nn.MaxPool1d
    # ------------------------------------------------------------------
    _orig_pool1d_init = ag_layers.Pool1d.__init__

    def _patched_pool1d_init(self, *args, **kwargs):
        _orig_pool1d_init(self, *args, **kwargs)
        if self.method == 'max':
            self._maxpool_module = nn.MaxPool1d(
                kernel_size=self.kernel_size, stride=self.stride
            )

    def _patched_pool1d_forward(self, x):
        input_size = x.shape[-1]
        output_size = (input_size + self.stride - 1) // self.stride
        pad_total = max(
            (output_size - 1) * self.stride + self.kernel_size - input_size, 0
        )
        if pad_total > 0:
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            x = F.pad(x, (pad_left, pad_right))
        if self.method == 'max':
            return self._maxpool_module(x)
        elif self.method in ['avg', 'mean']:
            return F.avg_pool1d(
                x, kernel_size=self.kernel_size, stride=self.stride
            )
        else:
            raise NotImplementedError(
                f"Pooling method {self.method} not implemented"
            )

    ag_layers.Pool1d.__init__ = _patched_pool1d_init
    ag_layers.Pool1d.forward = _patched_pool1d_forward

    # ------------------------------------------------------------------
    # SequenceEncoder: replace single shared self.pool with 7 separate
    # Pool1d instances (tangermeme hooks save per-module state, so reusing
    # one module for 7 calls causes shape mismatches in backward).
    # ------------------------------------------------------------------
    _orig_seqenc_init = SequenceEncoder.__init__

    def _patched_seqenc_init(self):
        _orig_seqenc_init(self)
        del self.pool
        self.pools = nn.ModuleList([
            ag_layers.Pool1d(kernel_size=2) for _ in range(7)
        ])

    def _patched_seqenc_forward(self, x):
        x = x.transpose(1, 2)  # (B, S, 4) -> (B, 4, S)
        intermediates = {}
        x = self.dna_embedder(x)
        intermediates['bin_size_1'] = x
        x = self.pools[0](x)

        for i, block in enumerate(self.down_blocks):
            if self.gradient_checkpointing and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            bin_size = self.bin_sizes[i]
            intermediates[f'bin_size_{bin_size}'] = x
            x = self.pools[i + 1](x)

        return x, intermediates

    SequenceEncoder.__init__ = _patched_seqenc_init
    SequenceEncoder.forward = _patched_seqenc_forward

    # ------------------------------------------------------------------
    # ConvBlock: layers.gelu() -> self._gelu_module()
    # ------------------------------------------------------------------
    _orig_convblock_init = ConvBlock.__init__

    def _patched_convblock_init(self, *args, **kwargs):
        _orig_convblock_init(self, *args, **kwargs)
        self._gelu_module = AGCustomGELU()

    def _patched_convblock_forward(self, x):
        return self.conv(self._gelu_module(self.norm(x)))

    ConvBlock.__init__ = _patched_convblock_init
    ConvBlock.forward = _patched_convblock_forward

    # ------------------------------------------------------------------
    # MLPBlock: F.relu() -> self._relu_module()
    # ------------------------------------------------------------------
    _orig_mlpblock_init = ag_attn.MLPBlock.__init__

    def _patched_mlpblock_init(self, *args, **kwargs):
        _orig_mlpblock_init(self, *args, **kwargs)
        self._relu_module = nn.ReLU()

    def _patched_mlpblock_forward(self, x):
        h = self.norm(x)
        h = self._relu_module(self.fc1(h))
        h = self.fc2(h)
        return self.final_norm(h)

    ag_attn.MLPBlock.__init__ = _patched_mlpblock_init
    ag_attn.MLPBlock.forward = _patched_mlpblock_forward

    # ------------------------------------------------------------------
    # AttentionBiasBlock: F.gelu() -> self._gelu_module()
    # ------------------------------------------------------------------
    _orig_attnbias_init = ag_attn.AttentionBiasBlock.__init__

    def _patched_attnbias_init(self, *args, **kwargs):
        _orig_attnbias_init(self, *args, **kwargs)
        self._gelu_module = nn.GELU()

    def _patched_attnbias_forward(self, x):
        h = self._gelu_module(self.norm(x))
        h = self.proj(h)
        h = torch.repeat_interleave(h, 16, dim=1)
        h = torch.repeat_interleave(h, 16, dim=2)
        return h.permute(0, 3, 1, 2)

    ag_attn.AttentionBiasBlock.__init__ = _patched_attnbias_init
    ag_attn.AttentionBiasBlock.forward = _patched_attnbias_forward

    # ------------------------------------------------------------------
    # PairMLPBlock: F.relu() -> self._relu_module()
    # ------------------------------------------------------------------
    _orig_pairmlp_init = ag_attn.PairMLPBlock.__init__

    def _patched_pairmlp_init(self, *args, **kwargs):
        _orig_pairmlp_init(self, *args, **kwargs)
        self._relu_module = nn.ReLU()

    def _patched_pairmlp_forward(self, x):
        h = self.norm(x)
        h = self.linear1(h)
        h = self._relu_module(h)
        h = self.linear2(h)
        return h

    ag_attn.PairMLPBlock.__init__ = _patched_pairmlp_init
    ag_attn.PairMLPBlock.forward = _patched_pairmlp_forward

    # ------------------------------------------------------------------
    # MHABlock: torch.tanh() -> self._tanh_module()
    #           F.softmax()   -> self._softmax_module()
    # ------------------------------------------------------------------
    _orig_mha_init = ag_attn.MHABlock.__init__

    def _patched_mha_init(self, *args, **kwargs):
        _orig_mha_init(self, *args, **kwargs)
        self._tanh_module = nn.Tanh()
        self._softmax_module = nn.Softmax(dim=-1)

    def _patched_mha_forward(self, x, attention_bias, compute_dtype=None):
        B, S, D = x.shape
        if compute_dtype is None:
            compute_dtype = x.dtype
        x = x.to(compute_dtype)

        h = self.norm(x)

        q = self.norm_q(self.q_proj(h).view(B, S, 8, 128))
        k = self.norm_k(self.k_proj(h).view(B, S, 1, 128))
        v = self.norm_v(self.v_proj(h).view(B, S, 1, 192))

        q = apply_rope(q, inplace=True)
        k = apply_rope(k, inplace=True)

        q_t = q.permute(0, 2, 1, 3)
        k_t = k.permute(0, 2, 1, 3)

        att = torch.matmul(q_t, k_t.transpose(-2, -1)).float()
        att = att / math.sqrt(128.0)

        if attention_bias is not None:
            att = att + attention_bias.float()

        logits_soft_cap = 5.0
        att = self._tanh_module(att / logits_soft_cap) * logits_soft_cap
        attn_weights = self._softmax_module(att)

        v_t = v.permute(0, 2, 1, 3)
        y = torch.matmul(attn_weights.to(compute_dtype), v_t).float()
        y = y.to(compute_dtype)
        y = y.permute(0, 2, 1, 3).reshape(B, S, -1)

        y = self.linear_embedding(y)
        return self.final_norm(y)

    ag_attn.MHABlock.__init__ = _patched_mha_init
    ag_attn.MHABlock.forward = _patched_mha_forward

    # ------------------------------------------------------------------
    # SequenceToPairBlock: F.gelu() -> self._gelu_module()
    # ------------------------------------------------------------------
    _orig_s2p_init = ag_attn.SequenceToPairBlock.__init__

    def _patched_s2p_init(self, *args, **kwargs):
        _orig_s2p_init(self, *args, **kwargs)
        self._gelu_module = nn.GELU()

    def _patched_s2p_forward(self, x):
        x_pooled = self.pool(x.transpose(1, 2)).transpose(1, 2)
        x_norm = self.norm_seq2pair(x_pooled)

        B, S_prime, _ = x_norm.shape

        q = self.linear_q(x_norm).view(B, S_prime, self.num_heads, self.head_dim)
        k = self.linear_k(x_norm).view(B, S_prime, self.num_heads, self.head_dim)

        range_vec = torch.arange(
            -S_prime, S_prime, device=x.device, dtype=torch.float32
        )
        pos_feat = _central_mask_features(
            torch.abs(range_vec), self.num_heads, _MAX_RELATIVE_DISTANCE // 16
        )
        sign = torch.sign(range_vec).unsqueeze(-1)
        pos_feat = torch.cat([pos_feat, sign * pos_feat], dim=-1)
        pos_feat = pos_feat.to(x.dtype)

        pos_encoding = self.linear_pos_features(pos_feat).view(
            2 * S_prime, self.num_heads, self.head_dim
        )

        term_q = torch.einsum('bqhc,phc->bhqp', q + self.q_r_bias, pos_encoding)
        term_k = torch.einsum('bkhc,phc->bhkp', k + self.k_r_bias, pos_encoding)

        rel_q_a = _shift(term_q, S_prime, S_prime)
        rel_k_a = _shift(term_k, S_prime, S_prime)

        rel_q_a = rel_q_a.permute(0, 2, 3, 1)
        rel_k_a = rel_k_a.permute(0, 3, 2, 1)

        a = torch.einsum('bqhc,bkhc->bqkh', q, k)
        a = a + 0.5 * (rel_q_a + rel_k_a)

        x_gelu = self._gelu_module(x_norm)
        y_q = self.linear_y_q(x_gelu)
        y_k = self.linear_y_k(x_gelu)

        pair_act = self.linear_pair(a) + y_q.unsqueeze(2) + y_k.unsqueeze(1)
        return pair_act

    ag_attn.SequenceToPairBlock.__init__ = _patched_s2p_init
    ag_attn.SequenceToPairBlock.forward = _patched_s2p_forward

    # ------------------------------------------------------------------
    # RowAttentionBlock: F.softmax() -> self._softmax_module()
    # ------------------------------------------------------------------
    _orig_rowattn_init = ag_attn.RowAttentionBlock.__init__

    def _patched_rowattn_init(self, *args, **kwargs):
        _orig_rowattn_init(self, *args, **kwargs)
        self._softmax_module = nn.Softmax(dim=-1)

    def _patched_rowattn_forward(self, x, compute_dtype=None):
        if compute_dtype is None:
            compute_dtype = x.dtype
        x = x.to(compute_dtype)

        h = self.norm(x)
        q = self.linear_q(h)
        k = self.linear_k(h)
        v = self.linear_v(h)

        scale = 1.0 / math.sqrt(128.0)
        attn = torch.einsum('bpqf,bpkf->bpqk', q, k).float() * scale
        attn = self._softmax_module(attn)

        out = torch.einsum(
            'bpqk,bpkf->bpqf', attn.to(compute_dtype), v
        ).float()
        return out.to(compute_dtype)

    ag_attn.RowAttentionBlock.__init__ = _patched_rowattn_init
    ag_attn.RowAttentionBlock.forward = _patched_rowattn_forward

    print('AlphaGenome patches applied (all functional activations -> nn.Module).')
