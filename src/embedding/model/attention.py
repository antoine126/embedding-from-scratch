"""
src/embedding/model/attention.py
==================================
Chapitre 6 — Mécanisme d'attention multi-têtes.

Contient :
  - scaled_dot_product_attention : implémentation pédagogique (fallback)
  - MultiHeadAttention           : attention multi-têtes bidirectionnelle

L'implémentation utilise F.scaled_dot_product_attention (Flash Attention
via PyTorch >= 2.0) quand disponible, et bascule sur l'implémentation
manuelle pour les versions antérieures.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as functionnal

from embedding.model.config import ModelConfig
from embedding.model.embeddings import RotaryPositionalEncoding


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Attention scalaire produit-points (Vaswani et al. 2017).

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        Q:         (B, n_heads, L_q, d_k)
        K:         (B, n_heads, L_k, d_k)
        V:         (B, n_heads, L_k, d_v)
        mask:      (B, 1, 1, L_k) — 1=token réel, 0=padding
        dropout_p: probabilité de dropout sur les poids d'attention

    Returns:
        output  : (B, n_heads, L_q, d_v)
        weights : (B, n_heads, L_q, L_k) — pour visualisation
    """
    d_k = Q.size(-1)
    scale = math.sqrt(d_k)

    # 1. Scores bruts : QK^T / sqrt(d_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (B, h, L_q, L_k)

    # 2. Masque de padding : exp(-inf) = 0 après softmax
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # 3. Softmax sur la dimension des clés
    weights = functionnal.softmax(scores, dim=-1)

    # Gérer les lignes entières à -inf (séquences de padding pur) -> NaN -> 0
    weights = torch.nan_to_num(weights, nan=0.0)

    # 4. Dropout optionnel
    if dropout_p > 0.0 and torch.is_grad_enabled():
        weights = functionnal.dropout(weights, p=dropout_p)

    # 5. Moyenne pondérée des valeurs
    output = torch.matmul(weights, V)  # (B, h, L_q, d_v)

    return output, weights


class MultiHeadAttention(nn.Module):
    """
    Attention multi-têtes bidirectionnelle pour encodeur.

    L'attention bidirectionnelle (sans masque causal) permet à chaque token
    d'attendre sur tous les autres — c'est le comportement voulu pour
    un encodeur d'embedding.

    Optimisations :
      - Projections QKV fusionnées en une seule matrice (3×d_model)
      - Flash Attention via F.scaled_dot_product_attention si PyTorch >= 2.0
      - Support optionnel de RoPE (Rotary Position Embedding)
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        assert config.d_model % config.n_heads == 0, (
            f"d_model ({config.d_model}) doit être divisible "
            f"par n_heads ({config.n_heads})"
        )
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.d_k = config.d_model // config.n_heads
        self.dropout_p = config.dropout

        # Projections QKV fusionnées pour l'efficacité
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=True)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=True)

        # RoPE : instancié uniquement si l'encodage positionnel est "rope"
        self.rope: RotaryPositionalEncoding | None = None
        if config.pos_encoding == "rope":
            self.rope = RotaryPositionalEncoding(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, L, d_model) — séquence d'entrée
            attention_mask: (B, L) — 1=token réel, 0=padding

        Returns:
            (B, L, d_model) — séquence après attention
        """
        B, L, d = x.shape

        # --- Projection QKV en une seule opération ---
        qkv = self.qkv_proj(x)  # (B, L, 3*d_model)
        Q, K, V = qkv.chunk(3, dim=-1)  # chacun (B, L, d_model)

        # --- Redimensionner pour les têtes : (B, L, d) -> (B, h, L, d_k) ---
        def to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        Q = to_heads(Q)
        K = to_heads(K)
        V = to_heads(V)

        # --- Appliquer RoPE sur Q et K si activé ---
        if self.rope is not None:
            Q = self.rope.apply_rope(Q, seq_len=L)
            K = self.rope.apply_rope(K, seq_len=L)

        # --- Construire le masque d'attention : (B, L) -> (B, 1, 1, L) ---
        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # --- Attention (Flash Attention si PyTorch >= 2.0) ---
        if hasattr(functionnal, "scaled_dot_product_attention"):
            bool_mask = attn_mask.bool() if attn_mask is not None else None
            attn_output = functionnal.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=bool_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        else:
            attn_output, _ = scaled_dot_product_attention(
                Q,
                K,
                V,
                mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            )

        # --- Recombiner les têtes : (B, h, L, d_k) -> (B, L, d_model) ---
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, d)

        return self.out_proj(attn_output)
