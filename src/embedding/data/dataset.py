"""
src/embedding/data/dataset.py
================================
Chapitres 4 et 12-13 — Datasets pour l'entraînement contrastif et MLM.

Datasets :
  - PairDataset    : paires (query, positive) pour MNR Loss (Ch. 13)
  - TripletDataset : triplets (anchor, positive, negative) pour Triplet Loss
  - MLMDataset     : corpus pour le pré-entraînement MLM (Ch. 12)

Format attendu des fichiers JSONL :
  PairDataset    : {"query": "...", "positive": "..."}
  TripletDataset : {"anchor": "...", "positive": "...", "negative": "..."}
  MLMDataset     : {"text": "..."}
"""
from __future__ import annotations

import json
import random
from collections.abc import Iterator
from pathlib import Path

from torch.utils.data import Dataset, IterableDataset


class PairDataset(Dataset):
    """
    Dataset de paires (query, positive) pour l'entraînement avec MNR Loss.

    Les autres exemples du batch servent de négatifs in-batch —
    c'est le principe fondamental de la Multiple Negative Ranking Loss.
    """

    def __init__(self, path: str | Path) -> None:
        self.samples: list[dict[str, str]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.samples[idx]


class TripletDataset(Dataset):
    """
    Dataset de triplets (anchor, positive, negative) pour Triplet Loss.

    Le négatif est fourni explicitement — contrairement à MNR où il est
    sélectionné dynamiquement parmi les autres exemples du batch.
    """

    def __init__(self, path: str | Path) -> None:
        self.samples: list[dict[str, str]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, str]:
        return self.samples[idx]


class MLMDataset(IterableDataset):
    """
    Dataset iterable pour le pré-entraînement MLM (Chapitre 12).

    Lit les textes depuis un fichier JSONL et les segmente en chunks
    de max_seq_len tokens. Les chunks sont mélangés par buffer
    pour éviter les corrélations entre exemples consécutifs.

    Le masquage (80/10/10) est appliqué dans MLMDataCollator.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: object,
        max_seq_len: int = 512,
        buffer_size: int = 10_000,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[dict[str, list[int]]]:
        buffer: list[dict[str, list[int]]] = []

        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text", line) if isinstance(data, dict) else line
                except json.JSONDecodeError:
                    text = line
                if not text:
                    continue

                # Tokeniser sans tokens spéciaux (ajoutés par le collator)
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                if hasattr(ids, "ids"):
                    ids = ids.ids  # tokenizers.Tokenizer

                # Segmenter en chunks de max_seq_len - 2 (pour [CLS] et [SEP])
                chunk_size = self.max_seq_len - 2
                for i in range(0, len(ids), chunk_size):
                    chunk = ids[i : i + chunk_size]
                    if len(chunk) < 10:  # Ignorer les chunks trop courts
                        continue
                    buffer.append({"input_ids": chunk})

                    if len(buffer) >= self.buffer_size:
                        random.shuffle(buffer)
                        yield from buffer
                        buffer = []

        if buffer:
            random.shuffle(buffer)
            yield from buffer
