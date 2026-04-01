"""
src/embedding/model/encoder.py
================================
Chapitre 8 — Modèle d'embedding Transformer encodeur complet.

Assemble toutes les briques des chapitres 5-7 :
  TokenEmbedding -> PositionalEncoding -> NxTransformerBlock
  -> LayerNorm finale -> Pooling -> normalisation L2

La normalisation L2 en sortie est essentielle pour que la similarité
cosinus soit bien définie et équivalente au produit scalaire.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functionnal

from embedding.model.config import ModelConfig
from embedding.model.embeddings import positional_encoding, TokenEmbedding
from embedding.model.layers import Pooling, TransformerBlock


class EmbeddingModel(nn.Module):
    """
    Modèle d'embedding Transformer encodeur.

    Architecture (Pre-LN) :
        x = TokenEmbedding(input_ids) + PositionalEncoding(x)
        x = Dropout(x)
        for block in TransformerBlocks:
            x = block(x, attention_mask)
        x = LayerNorm(x)
        emb = Pooling(x, attention_mask)
        emb = L2_normalize(emb)

    Usage :
        config = ModelConfig(d_model=768, n_heads=12, n_layers=6)
        model  = EmbeddingModel(config)
        emb    = model(input_ids, attention_mask)  # (B, 768), normalisés L2
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = TokenEmbedding(config)
        self.position_encoding = positional_encoding(config)
        self.embedding_dropout = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.post_norm = nn.LayerNorm(config.d_model, eps=1e-12)
        self.pooling = Pooling(config)

        self._init_weights()
        print(
            f"EmbeddingModel initialisé : "
            f"{self.n_parameters / 1e6:.1f}M paramètres"
        )

    def _init_weights(self) -> None:
        """
        Initialisation des poids selon le schéma standard (Radford 2019).
        std=0.02 pour les linéaires et embeddings.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Passe forward complète.

        Args:
            input_ids:      (B, L) — identifiants de tokens
            attention_mask: (B, L) — 1=token réel, 0=padding

        Returns:
            (B, d_model) — embeddings normalisés L2
        """
        x = self.token_embedding(input_ids)  # (B, L, d_model)
        x = self.position_encoding(x)  # (B, L, d_model)
        x = self.embedding_dropout(x)

        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.post_norm(x)  # normalisation finale de la séquence
        emb = self.pooling(x, attention_mask)  # (B, d_model)

        # Normalisation L2 : sim cosinus = produit scalaire
        return functionnal.normalize(emb, p=2, dim=-1)

    @property
    def n_parameters(self) -> int:
        """Nombre de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def encode(
        self,
        texts: list[str],
        tokenizer: object,
        batch_size: int = 64,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Encode une liste de textes en embeddings (inférence).

        Args:
            texts:      Liste de chaînes à encoder.
            tokenizer:  Tokenizer HuggingFace (AutoTokenizer) ou
                        tokenizers.Tokenizer (HuggingFace tokenizers).
            batch_size: Taille du batch d'inférence.
            device:     Device cible ("cpu", "cuda", "mps").

        Returns:
            (N, d_model) — embeddings normalisés L2 sur CPU.
        """
        self.eval()
        all_embeddings: list[torch.Tensor] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]

            # Compatibilité HuggingFace tokenizers et transformers
            if hasattr(tokenizer, "encode_batch"):
                # tokenizers.Tokenizer
                enc = tokenizer.encode_batch(batch)
                input_ids = torch.tensor(
                    [e.ids for e in enc], dtype=torch.long, device=device
                )
                mask = torch.tensor(
                    [e.attention_mask for e in enc],
                    dtype=torch.long,
                    device=device,
                )
            else:
                # transformers.AutoTokenizer
                enc = tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_seq_len,
                    return_tensors="pt",
                )
                input_ids = enc["input_ids"].to(device)
                mask = enc["attention_mask"].to(device)

            emb = self(input_ids, mask)
            all_embeddings.append(emb.cpu())

        return torch.cat(all_embeddings, dim=0)
