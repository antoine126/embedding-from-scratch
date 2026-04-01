"""
src/embedding/rag/reranker.py
================================
Chapitre 20 — Re-ranking avec cross-encoder.

Le re-ranking est une étape post-retrieval qui réévalue les K candidats
récupérés par le retriever (bi-encoder) avec un modèle plus précis
mais plus lent (cross-encoder).

Architecture :
  - Bi-encoder (retriever) : encode query et document séparément
    -> efficace pour indexer des millions de documents
  - Cross-encoder (reranker) : encode (query, document) ensemble
    -> plus précis mais trop lent pour de gros corpus

Pipeline à deux étapes :
  1. Retrieval rapide : top-100 avec FAISS
  2. Re-ranking précis : top-10 avec cross-encoder
"""
from __future__ import annotations

from dataclasses import dataclass

from embedding.rag.retriever import SearchResult


@dataclass
class CrossEncoderReranker:
    """
    Re-ranker basé sur un cross-encoder HuggingFace.

    Utilise un modèle pré-entraîné de cross-encoder
    (ex: cross-encoder/ms-marco-MiniLM-L-6-v2) pour scorer
    les paires (query, passage) directement.

    Args:
        model_name: Nom du modèle cross-encoder sur HuggingFace Hub.
        device:     Device pour l'inférence.
        batch_size: Taille de batch pour le scoring.
    """

    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: str = "cpu"
    batch_size: int = 32

    def __post_init__(self) -> None:
        self._model = None

    def _load_model(self) -> None:
        """Charge le modèle cross-encoder (lazy loading)."""
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers est requis pour le reranker. "
                "Installez-le avec : uv add sentence-transformers"
            ) from e

        if self._model is None:
            self._model = CrossEncoder(self.model_name, device=self.device)

    def rerank(
        self,
        query: str,
        candidates: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Re-classe les candidats avec le cross-encoder.

        Args:
            query:      Requête originale.
            candidates: Résultats du retriever à re-classer.
            top_k:      Nombre de résultats à retourner (None = tous).

        Returns:
            Candidats re-classés par score cross-encoder décroissant.
        """
        if not candidates:
            return candidates

        self._load_model()

        # Paires (query, passage) pour le cross-encoder
        pairs = [(query, result.chunk.text) for result in candidates]

        # Scoring par batch
        scores = self._model.predict(  # type: ignore
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        # Trier par score décroissant
        scored = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        if top_k is not None:
            scored = scored[:top_k]

        return [
            SearchResult(
                chunk=result.chunk,
                score=float(score),
                rank=rank,
            )
            for rank, (result, score) in enumerate(scored, start=1)
        ]
