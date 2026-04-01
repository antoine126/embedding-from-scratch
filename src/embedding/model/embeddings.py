"""
src/embedding/model/embeddings.py
===================================
Chapitre 5 — Couche d'embedding d'entrée.

Contient :
  - TokenEmbedding   : table de lookup (B, L) -> (B, L, d)
  - SinusoidalPositionalEncoding : PE non-paramétrique (Vaswani 2017)
  - LearnedPositionalEncoding    : PE paramétrique (BERT-style)
  - RotaryPositionalEncoding     : RoPE (Su et al. 2024)
  - PositionalEncoding           : factory sélectionnant selon config.pos_encoding
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from embedding.model.config import ModelConfig


class TokenEmbedding(nn.Module):
    """
    Table d'embedding de tokens.

    Convertit une séquence d'identifiants (B, L) en vecteurs (B, L, d_model).
    Le token [PAD] (id=0) est mappé au vecteur nul avec gradient nul.
    Les vecteurs sont mis à l'échelle par sqrt(d_model) (Vaswani 2017, §3.4).
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.scale = math.sqrt(config.d_model)

        self.embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=0,  # [PAD] -> vecteur nul, gradient nul
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids : (B, L) -> (B, L, d_model)
        return self.embedding(input_ids) * self.scale


class SinusoidalPositionalEncoding(nn.Module):
    """
    Encodage positionnel sinusoïdal (Vaswani et al. 2017).

    Non paramétrique : aucun paramètre appris.
    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Avantage : peut généraliser à des longueurs non vues à l'entraînement.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.d_model
        max_len = config.max_seq_len

        pe = torch.zeros(max_len, d)
        position = torch.arange(max_len).unsqueeze(1).float()  # (L_max, 1)
        div_term = torch.exp(
            torch.arange(0, d, 2).float() * (-math.log(10000.0) / d)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # dimensions paires
        pe[:, 1::2] = torch.cos(position * div_term)  # dimensions impaires
        pe = pe.unsqueeze(0)  # (1, L_max, d) — broadcastable sur le batch

        # register_buffer : déplacé avec .to(device) mais non entraînable
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L, d_model)
        return x + self.pe[:, : x.size(1)]  # type: ignore[index]


class LearnedPositionalEncoding(nn.Module):
    """
    Encodage positionnel appris (style BERT).

    La matrice de position est un nn.Embedding mis à jour par backprop.
    Plus flexible que les encodages sinusoïdaux mais limité à max_seq_len.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.pe = nn.Embedding(config.max_seq_len, config.d_model)
        nn.init.normal_(self.pe.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L, d_model)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)  # (1, L)
        return x + self.pe(positions)  # (B, L, d_model)


class RotaryPositionalEncoding(nn.Module):
    """
    RoPE : Rotary Position Embedding (Su et al. 2024).

    Encode la position par rotation dans l'espace des queries/keys plutôt
    qu'en ajoutant un vecteur à l'entrée. Propriété principale : la similarité
    entre Q_i et K_j ne dépend que de la distance relative (i - j).

    Cette classe précalcule les cosinus/sinus pour les réutiliser dans
    MultiHeadAttention via apply_rope().
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d_head = config.d_head
        max_len = config.max_seq_len

        # theta_i = 10000^(-2i/d_head)
        theta = 1.0 / (
            10000.0 ** (torch.arange(0, d_head, 2).float() / d_head)
        )
        positions = torch.arange(max_len).float()
        freqs = torch.outer(positions, theta)  # (L_max, d_head/2)

        self.register_buffer("cos", freqs.cos())  # (L_max, d_head/2)
        self.register_buffer("sin", freqs.sin())  # (L_max, d_head/2)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotation de 90° : (x1, x2) -> (-x2, x1)."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Applique RoPE au tenseur x (queries ou keys).

        Args:
            x:       (B, n_heads, L, d_head)
            seq_len: longueur de la séquence

        Returns:
            (B, n_heads, L, d_head) avec position encodée par rotation
        """
        cos = self.cos[:seq_len].unsqueeze(0).unsqueeze(0)  # (1, 1, L, d/2)
        sin = self.sin[:seq_len].unsqueeze(0).unsqueeze(0)

        # Répéter pour couvrir toutes les dimensions de d_head
        cos = cos.repeat_interleave(2, dim=-1)  # (1, 1, L, d_head)
        sin = sin.repeat_interleave(2, dim=-1)

        return x * cos + self._rotate_half(x) * sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RoPE est appliqué dans MultiHeadAttention, pas ici.
        # Cette méthode est un no-op pour compatibilité d'interface.
        return x


def positional_encoding(config: ModelConfig) -> nn.Module:
    """
    Factory : retourne l'encodage positionnel configuré dans ModelConfig.

    config.pos_encoding : "sinusoidal" | "learned" | "rope"
    """
    kinds = {
        "sinusoidal": SinusoidalPositionalEncoding,
        "learned": LearnedPositionalEncoding,
        "rope": RotaryPositionalEncoding,
    }
    if config.pos_encoding not in kinds:
        raise ValueError(
            f"pos_encoding inconnu : {config.pos_encoding!r}. "
            f"Valeurs valides : {list(kinds)}"
        )
    return kinds[config.pos_encoding](config)
