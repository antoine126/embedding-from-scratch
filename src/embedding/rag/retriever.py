"""
src/embedding/rag/retriever.py
==================================
Chapitre 18 — Retrieval avec index vectoriel FAISS.

Le retriever encode les documents dans un espace vectoriel et
retrouve les plus similaires à une requête via FAISS.

Deux classes :
  - DenseRetriever : recherche dense par similarité cosinus (embedding)
  - BM25Retriever  : recherche lexicale BM25 (fréquence de termes)

Ces deux classes sont combinées dans HybridRetriever (rag/hybrid.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn

from embedding.rag.chunker import Chunk


class SearchResult(NamedTuple):
    """Résultat d'une recherche."""

    chunk: Chunk
    score: float
    rank: int


@dataclass
class DenseRetriever:
    """
    Retriever dense basé sur les embeddings et FAISS.

    Utilise IndexFlatIP (produit scalaire) sur des embeddings normalisés L2,
    ce qui est équivalent à la similarité cosinus.

    Args:
        model:     Modèle d'embedding (EmbeddingModel ou compatible HF).
        tokenizer: Tokenizer correspondant au modèle.
        device:    Device pour l'encodage.
        batch_size: Taille de batch pour l'encodage du corpus.
    """

    model: nn.Module
    tokenizer: object
    device: str = "cpu"
    batch_size: int = 256

    def __post_init__(self) -> None:
        self._index = None
        self._chunks: list[Chunk] = []

    def index_documents(self, chunks: list[Chunk]) -> None:
        """
        Encode et indexe une liste de chunks.

        Args:
            chunks: Passages à indexer.
        """
        try:
            import faiss
        except ImportError as e:
            raise ImportError(
                "faiss est requis pour le retriever dense. "
                "Installez-le avec : uv add faiss-cpu"
            ) from e

        self._chunks = chunks
        texts = [c.text for c in chunks]
        embeddings = self._encode(texts)

        d = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(d)
        self._index.add(embeddings)  # type: ignore

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        Retrouve les top_k passages les plus similaires à la requête.

        Args:
            query: Requête en texte libre.
            top_k: Nombre de résultats à retourner.

        Returns:
            Liste de SearchResult ordonnés par score décroissant.
        """
        if self._index is None:
            raise RuntimeError("Appeler index_documents() avant search().")

        query_emb = self._encode([query])  # (1, d)
        scores, indices = self._index.search(query_emb, top_k)  # type: ignore

        results = []
        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]), start=1
        ):
            if idx < 0:  # FAISS retourne -1 si moins de top_k résultats
                break
            results.append(
                SearchResult(
                    chunk=self._chunks[idx],
                    score=float(score),
                    rank=rank,
                )
            )
        return results

    def _encode(self, texts: list[str]) -> np.ndarray:
        """Encode des textes en embeddings numpy float32."""
        self.model.eval()
        embeddings: list[torch.Tensor] = []

        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]

                if hasattr(self.tokenizer, "encode_batch"):
                    enc = self.tokenizer.encode_batch(batch)
                    input_ids = torch.tensor(
                        [e.ids for e in enc], dtype=torch.long, device=self.device
                    )
                    mask = torch.tensor(
                        [e.attention_mask for e in enc],
                        dtype=torch.long,
                        device=self.device,
                    )
                else:
                    enc = self.tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                    )
                    input_ids = enc["input_ids"].to(self.device)
                    mask = enc["attention_mask"].to(self.device)

                emb = self.model(input_ids, mask)
                embeddings.append(emb.cpu())

        return torch.cat(embeddings, dim=0).numpy().astype(np.float32)

    def save_index(self, path: str | Path) -> None:
        """Sauvegarde l'index FAISS et les chunks sur disque."""
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))

        chunks_data = [
            {
                "text": c.text,
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "metadata": c.metadata,
            }
            for c in self._chunks
        ]
        with open(path / "chunks.jsonl", "w", encoding="utf-8") as f:
            for chunk in chunks_data:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    def load_index(self, path: str | Path) -> None:
        """Charge l'index FAISS et les chunks depuis le disque."""
        import faiss

        path = Path(path)
        self._index = faiss.read_index(str(path / "index.faiss"))

        self._chunks = []
        with open(path / "chunks.jsonl", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                self._chunks.append(
                    Chunk(
                        text=d["text"],
                        doc_id=d["doc_id"],
                        chunk_id=d["chunk_id"],
                        metadata=d.get("metadata", {}),
                    )
                )


@dataclass
class BM25Retriever:
    """
    Retriever lexical BM25 (Best Match 25).

    BM25 est la baseline standard en recherche d'information :
    score basé sur la fréquence des termes de la requête dans le document,
    normalisé par la longueur du document.

    Complémentaire à la recherche dense : BM25 excelle sur les requêtes
    exactes (noms propres, codes, termes techniques rares)
    où les embeddings peuvent manquer de précision.
    """

    def __post_init__(self) -> None:
        self._bm25 = None
        self._chunks: list[Chunk] = []

    def index_documents(self, chunks: list[Chunk]) -> None:
        """Indexe les chunks avec BM25."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError(
                "rank-bm25 est requis. Installez-le avec : uv add rank-bm25"
            ) from e

        self._chunks = chunks
        tokenized = [c.text.lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """
        Retrouve les top_k passages avec BM25.

        Args:
            query: Requête en texte libre.
            top_k: Nombre de résultats.

        Returns:
            Liste de SearchResult ordonnés par score BM25 décroissant.
        """
        if self._bm25 is None:
            raise RuntimeError("Appeler index_documents() avant search().")

        query_tokens = query.lower().split()
        scores = self._bm25.get_scores(query_tokens)

        # Top-K indices par score décroissant
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            SearchResult(
                chunk=self._chunks[idx],
                score=float(scores[idx]),
                rank=rank,
            )
            for rank, idx in enumerate(top_indices, start=1)
        ]
