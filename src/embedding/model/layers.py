"""
src/embedding/model/layers.py
================================
Chapitre 7 — Autres couches du Transformer encodeur.

Contient :
  - FeedForward      : réseau position-wise avec activations configurables
                       (GELU, ReLU, SwiGLU, GeGLU)
  - TransformerBlock : bloc complet Pre-LN (MHA + FFN + résidus)
  - Pooling          : agrégation séquence -> vecteur
                       (mean, cls, weighted_mean)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functionnal

from embedding.model.attention import MultiHeadAttention
from embedding.model.config import ModelConfig


class FeedForward(nn.Module):
    """
    Réseau feed-forward position-wise.

    Traite chaque token indépendamment : (B, L, d) -> (B, L, d).
    Supporte les variantes GLU (SwiGLU, GeGLU) utilisées dans les LLM modernes.

    Activations disponibles (config.activation) :
      - "gelu"   : FFN standard, activation GELU (BERT, RoBERTa)
      - "relu"   : FFN standard, activation ReLU (Transformer original)
      - "swiglu" : SwiGLU (LLaMA, PaLM) — deux matrices d'entrée
      - "geglu"  : GeGLU — variante GELU des GLU

    Note sur les variantes GLU : d_ff est réduit à 2/3 pour compenser le
    paramètre supplémentaire (W3) et garder le même nombre de paramètres.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        d = config.d_model
        d_ff = config.d_ff
        act = config.activation

        self.activation_type = act
        self.dropout = nn.Dropout(config.dropout)

        if act in ("swiglu", "geglu"):
            # Réduction de d_ff pour compenser W3
            d_ff_glu = int(d_ff * 2 / 3)
            self.W1 = nn.Linear(d, d_ff_glu, bias=False)  # branche gate
            self.W2 = nn.Linear(d_ff_glu, d, bias=False)  # projection sortie
            self.W3 = nn.Linear(d, d_ff_glu, bias=False)  # branche value
        else:
            self.W1 = nn.Linear(d, d_ff, bias=True)
            self.W2 = nn.Linear(d_ff, d, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : (B, L, d_model) -> (B, L, d_model)
        if self.activation_type == "gelu":
            return self.dropout(self.W2(functionnal.gelu(self.W1(x))))

        if self.activation_type == "relu":
            return self.dropout(self.W2(functionnal.relu(self.W1(x))))

        if self.activation_type == "swiglu":
            # SwiGLU(x) = SiLU(W1·x) ⊙ W3·x
            gate = functionnal.silu(self.W1(x))
            value = self.W3(x)
            return self.dropout(self.W2(gate * value))

        if self.activation_type == "geglu":
            # GeGLU(x) = GELU(W1·x) ⊙ W3·x
            gate = functionnal.gelu(self.W1(x))
            value = self.W3(x)
            return self.dropout(self.W2(gate * value))

        raise ValueError(f"Activation inconnue : {self.activation_type!r}")


class TransformerBlock(nn.Module):
    """
    Bloc Transformer encodeur avec Pre-Layer Normalization.

    Architecture Pre-LN (contrairement au Post-LN de Vaswani 2017) :
      x = x + Attention(LayerNorm(x))
      x = x + FFN(LayerNorm(x))

    Pre-LN est plus stable à l'entraînement : les gradients restent
    dans une plage raisonnable même sans warmup prolongé.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model, eps=1e-12)
        self.attn = MultiHeadAttention(config)
        self.norm2 = nn.LayerNorm(config.d_model, eps=1e-12)
        self.ffn = FeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Args:
            x:              (B, L, d_model)
            attention_mask: (B, L) — 1=token, 0=padding

        Returns:
            (B, L, d_model)
        """
        # Sous-couche 1 : attention avec connexion résiduelle (Pre-LN)
        x = x + self.attn(self.norm1(x), attention_mask)

        # Sous-couche 2 : FFN avec connexion résiduelle (Pre-LN)
        x = x + self.ffn(self.norm2(x))

        return x


class Pooling(nn.Module):
    """
    Agrégation de la séquence en un vecteur unique.

    Stratégies disponibles (config.pooling) :
      - "cls"           : vecteur du token [CLS] (position 0)
      - "mean"          : moyenne des tokens réels (masque appliqué)
      - "weighted_mean" : moyenne pondérée par position (dernier token = poids max)

    Mean pooling est généralement supérieur à CLS pour les encodeurs
    entraînés avec MNR Loss, car tous les tokens contribuent à l'embedding.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.strategy = config.pooling

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states:  (B, L, d_model) — sorties du dernier bloc
            attention_mask: (B, L) — 1=token réel, 0=padding

        Returns:
            (B, d_model) — embedding de la séquence
        """
        if self.strategy == "cls":
            return self._cls_pooling(hidden_states)
        if self.strategy == "mean":
            return self._mean_pooling(hidden_states, attention_mask)
        if self.strategy == "weighted_mean":
            return self._weighted_mean_pooling(hidden_states, attention_mask)
        raise ValueError(f"Stratégie de pooling inconnue : {self.strategy!r}")

    @staticmethod
    def _cls_pooling(hidden_states: torch.Tensor) -> torch.Tensor:
        """Vecteur du token [CLS] (position 0)."""
        return hidden_states[:, 0, :]  # (B, d)

    @staticmethod
    def _mean_pooling(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Moyenne des vrais tokens (padding exclu)."""
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        sum_embeddings = (hidden_states * mask).sum(dim=1)  # (B, d)
        sum_mask = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
        return sum_embeddings / sum_mask  # (B, d)

    @staticmethod
    def _weighted_mean_pooling(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Moyenne pondérée par position (tokens tardifs = poids plus élevé)."""
        b_, l_, _ = hidden_states.shape
        mask = attention_mask.float()

        # Poids linéaires de 1 à L, remis à 0 pour le padding
        positions = torch.arange(1, l_ + 1, device=hidden_states.device)
        positions = positions.unsqueeze(0).expand(b_, -1)  # (B, L)
        weights = positions * mask  # (B, L)

        # Normalisation
        weights = weights / weights.sum(dim=1, keepdim=True).clamp(min=1e-9)
        weights = weights.unsqueeze(-1)  # (B, L, 1)

        return (hidden_states * weights).sum(dim=1)  # (B, d)
