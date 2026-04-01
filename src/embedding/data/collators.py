"""
src/embedding/data/collators.py
==================================
Chapitres 4 et 12 — Collators pour assembler les batches.

Collators :
  - PairCollator    : paires (query, positive) -> tenseurs paddés (Ch. 4, 13)
  - TripletCollator : triplets -> tenseurs paddés (Ch. 9)
  - MLMDataCollator : corpus -> masquage 80/10/10 pour MLM (Ch. 12)

Un collator est la fonction passée comme collate_fn au DataLoader.
Il assemble une liste d'exemples en un batch homogène avec padding dynamique.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from tokenizers import Tokenizer


@dataclass
class PairCollator:
    """
    Collate des paires (query, positive) pour la MNR Loss.

    Retourne deux séquences paddées dynamiquement avec leurs masques.
    Le padding est fait à la longueur maximale du batch (pas de padding fixe).
    """

    tokenizer: Tokenizer
    max_length: int = 512

    def __call__(
        self, batch: list[dict[str, str]]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            batch: liste de dicts {"query": str, "positive": str}

        Returns:
            dict avec :
              query_input_ids      : (B, L_q)
              query_attention_mask : (B, L_q)
              pos_input_ids        : (B, L_p)
              pos_attention_mask   : (B, L_p)
        """
        queries = [item["query"] for item in batch]
        positives = [item["positive"] for item in batch]

        q_enc = self._encode_batch(queries)
        p_enc = self._encode_batch(positives)

        return {
            "query_input_ids": q_enc["input_ids"],
            "query_attention_mask": q_enc["attention_mask"],
            "pos_input_ids": p_enc["input_ids"],
            "pos_attention_mask": p_enc["attention_mask"],
        }

    def _encode_batch(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Encode et padde un batch de textes."""
        pad_id = self.tokenizer.token_to_id("[PAD]")
        self.tokenizer.enable_padding(
            pad_id=pad_id,
            pad_token="[PAD]",
            length=None,  # Padding dynamique à la longueur max du batch
        )
        self.tokenizer.enable_truncation(max_length=self.max_length)

        encodings = self.tokenizer.encode_batch(texts)

        input_ids = torch.tensor(
            [enc.ids for enc in encodings], dtype=torch.long
        )
        attention_mask = torch.tensor(
            [enc.attention_mask for enc in encodings], dtype=torch.long
        )
        return {"input_ids": input_ids, "attention_mask": attention_mask}


@dataclass
class TripletCollator:
    """
    Collate des triplets (anchor, positive, negative) pour la Triplet Loss.
    """

    tokenizer: Tokenizer
    max_length: int = 512

    def __call__(
        self, batch: list[dict[str, str]]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            batch: liste de dicts {"anchor": str, "positive": str, "negative": str}

        Returns:
            dict avec anchor_*, pos_*, neg_* (input_ids + attention_mask)
        """
        anchors = [item["anchor"] for item in batch]
        positives = [item["positive"] for item in batch]
        negatives = [item["negative"] for item in batch]

        pad_id = self.tokenizer.token_to_id("[PAD]")
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token="[PAD]")
        self.tokenizer.enable_truncation(max_length=self.max_length)

        def encode(texts: list[str]) -> dict[str, torch.Tensor]:
            enc = self.tokenizer.encode_batch(texts)
            return {
                "input_ids": torch.tensor([e.ids for e in enc], dtype=torch.long),
                "attention_mask": torch.tensor(
                    [e.attention_mask for e in enc], dtype=torch.long
                ),
            }

        a = encode(anchors)
        p = encode(positives)
        n = encode(negatives)

        return {
            "anchor_input_ids": a["input_ids"],
            "anchor_attention_mask": a["attention_mask"],
            "pos_input_ids": p["input_ids"],
            "pos_attention_mask": p["attention_mask"],
            "neg_input_ids": n["input_ids"],
            "neg_attention_mask": n["attention_mask"],
        }


@dataclass
class MLMDataCollator:
    """
    Collator pour le pré-entraînement MLM (Chapitre 12).

    Applique la stratégie de masquage originale de BERT (Devlin et al. 2019) :
      - 15% des tokens sont sélectionnés pour la prédiction
      - Parmi eux :
        - 80% sont remplacés par [MASK]
        - 10% sont remplacés par un token aléatoire
        - 10% restent inchangés
      - Les positions non masquées ont label = -100 (ignorées par CrossEntropy)

    Args:
        mask_token_id : ID du token [MASK] dans le vocabulaire
        vocab_size    : taille du vocabulaire (pour les tokens aléatoires)
        mlm_probability : proportion de tokens à masquer (défaut 15%)
        pad_token_id  : ID du token [PAD] (défaut 3)
    """

    mask_token_id: int
    vocab_size: int
    mlm_probability: float = 0.15
    pad_token_id: int = 3

    def __call__(
        self, batch: list[dict[str, list[int]]]
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            batch: liste de dicts {"input_ids": [int, ...]}

        Returns:
            dict avec :
              input_ids      : (B, L) — séquence avec masques
              attention_mask : (B, L) — 1=token, 0=padding
              labels         : (B, L) — token original ou -100
        """
        # Padding dynamique à la longueur max du batch
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids_list = []
        attention_mask_list = []

        for item in batch:
            ids = item["input_ids"]
            pad_len = max_len - len(ids)
            input_ids_list.append(ids + [self.pad_token_id] * pad_len)
            attention_mask_list.append([1] * len(ids) + [0] * pad_len)

        input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long)

        # Masquage 80/10/10
        masked_input_ids, labels = self._apply_mlm_masking(
            input_ids, attention_mask
        )

        return {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _apply_mlm_masking(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Applique le masquage MLM selon la stratégie BERT 80/10/10.

        Returns:
            masked_input_ids : input_ids avec certains tokens remplacés
            labels           : token originals (-100 pour les non-masqués)
        """
        labels = input_ids.clone()
        masked_input_ids = input_ids.clone()

        # Probabilité de masquage appliquée uniquement sur les vrais tokens
        probability_matrix = torch.full(input_ids.shape, self.mlm_probability)
        probability_matrix[attention_mask == 0] = 0.0  # Pas de masquage sur padding

        # Sélection des positions à masquer
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Les positions non masquées ont label = -100 (ignorées par la loss)
        labels[~masked_indices] = -100

        # 80% : remplacer par [MASK]
        replace_with_mask = (
            torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        )
        masked_input_ids[replace_with_mask] = self.mask_token_id

        # 10% : remplacer par un token aléatoire
        replace_with_random = (
            torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool()
            & masked_indices
            & ~replace_with_mask
        )
        random_tokens = torch.randint(
            low=5,  # Éviter les tokens spéciaux (IDs 0-4)
            high=self.vocab_size,
            size=input_ids.shape,
            dtype=torch.long,
        )
        masked_input_ids[replace_with_random] = random_tokens[replace_with_random]

        # 10% : laisser inchangé (déjà fait par défaut)

        return masked_input_ids, labels
