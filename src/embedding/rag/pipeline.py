"""
src/embedding/rag/pipeline.py
================================
Chapitres 19-20 — Pipeline RAG de production avec recherche hybride.

Le pipeline complet combine :
  1. Recherche hybride : dense (embeddings) + lexicale (BM25)
     Fusion des scores par Reciprocal Rank Fusion (RRF)
  2. Re-ranking : cross-encoder pour re-classer les meilleurs candidats
  3. Assemblage du contexte : prompt LLM avec les passages pertinents

Reciprocal Rank Fusion (RRF) :
  score_rrf(d) = Σ_r 1 / (k + rank_r(d))
  k=60 est la constante standard (Cormack et al. 2009).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from embedding.rag.chunker import Chunk
from embedding.rag.reranker import CrossEncoderReranker
from embedding.rag.retriever import BM25Retriever, DenseRetriever, SearchResult


@dataclass
class HybridRetriever:
    """
    Recherche hybride : dense + lexicale avec Reciprocal Rank Fusion.

    La fusion RRF combine les rankings sans normalisation des scores,
    ce qui évite les problèmes de calibration entre les deux systèmes.

    Args:
        dense_retriever: Retriever dense (DenseRetriever).
        bm25_retriever:  Retriever lexical (BM25Retriever).
        rrf_k:           Constante RRF (défaut 60).
        dense_weight:    Poids du retriever dense dans la fusion (0-1).
    """

    dense_retriever: DenseRetriever
    bm25_retriever: BM25Retriever
    rrf_k: int = 60
    dense_weight: float = 0.5

    def index_documents(self, chunks: list[Chunk]) -> None:
        """Indexe les chunks dans les deux retrievers."""
        self.dense_retriever.index_documents(chunks)
        self.bm25_retriever.index_documents(chunks)

    def search(
        self,
        query: str,
        top_k: int = 20,
        n_candidates: int = 100,
    ) -> list[SearchResult]:
        """
        Recherche hybride avec fusion RRF.

        Args:
            query:        Requête en texte libre.
            top_k:        Nombre de résultats finaux.
            n_candidates: Nombre de candidats par retriever avant fusion.

        Returns:
            Top-k résultats fusionnés, ordonnés par score RRF.
        """
        # Recherches indépendantes
        dense_results = self.dense_retriever.search(query, top_k=n_candidates)
        bm25_results = self.bm25_retriever.search(query, top_k=n_candidates)

        # Fusion par Reciprocal Rank Fusion
        return self._reciprocal_rank_fusion(dense_results, bm25_results, top_k)

    def _reciprocal_rank_fusion(
        self,
        dense_results: list[SearchResult],
        bm25_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """
        Fusionne deux rankings avec Reciprocal Rank Fusion (RRF).

        score_rrf(d) = Σ_r weight_r / (k + rank_r(d))
        """
        # Accumuler les scores RRF par identifiant de chunk
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, Chunk] = {}

        for rank, result in enumerate(dense_results, start=1):
            chunk_key = f"{result.chunk.doc_id}_{result.chunk.chunk_id}"
            rrf_scores[chunk_key] = rrf_scores.get(chunk_key, 0.0)
            rrf_scores[chunk_key] += self.dense_weight / (self.rrf_k + rank)
            chunk_map[chunk_key] = result.chunk

        lexical_weight = 1.0 - self.dense_weight
        for rank, result in enumerate(bm25_results, start=1):
            chunk_key = f"{result.chunk.doc_id}_{result.chunk.chunk_id}"
            rrf_scores[chunk_key] = rrf_scores.get(chunk_key, 0.0)
            rrf_scores[chunk_key] += lexical_weight / (self.rrf_k + rank)
            chunk_map[chunk_key] = result.chunk

        # Trier par score RRF décroissant
        sorted_keys = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

        return [
            SearchResult(
                chunk=chunk_map[key],
                score=rrf_scores[key],
                rank=rank,
            )
            for rank, key in enumerate(sorted_keys[:top_k], start=1)
        ]


@dataclass
class RAGPipeline:
    """
    Pipeline RAG complet : retrieval hybride + re-ranking + génération.

    Usage :
        pipeline = RAGPipeline(
            retriever=HybridRetriever(dense, bm25),
            reranker=CrossEncoderReranker(),
        )
        pipeline.index_documents(chunks)
        response = pipeline.query("Qu'est-ce qu'un embedding ?")

    Args:
        retriever:        HybridRetriever ou DenseRetriever.
        reranker:         CrossEncoderReranker (optionnel).
        n_retrieval:      Nombre de candidats pour le retrieval.
        n_final:          Nombre de passages dans le contexte final.
        context_template: Template du prompt LLM.
    """

    retriever: HybridRetriever | DenseRetriever
    reranker: Optional[CrossEncoderReranker] = None
    n_retrieval: int = 20
    n_final: int = 5
    context_template: str = (
        "Contexte :\n{context}\n\nQuestion : {query}\n\nRéponse :"
    )

    def index_documents(self, chunks: list[Chunk]) -> None:
        """Indexe les chunks dans le retriever."""
        self.retriever.index_documents(chunks)

    def retrieve(self, query: str) -> list[SearchResult]:
        """
        Retrouve les passages pertinents pour une requête.

        Pipeline :
          1. Retrieval (dense ou hybride) -> n_retrieval candidats
          2. Re-ranking (optionnel) -> n_final passages

        Args:
            query: Requête en texte libre.

        Returns:
            Liste des n_final passages les plus pertinents.
        """
        # Étape 1 : Retrieval
        candidates = self.retriever.search(query, top_k=self.n_retrieval)

        # Étape 2 : Re-ranking (optionnel)
        if self.reranker is not None:
            candidates = self.reranker.rerank(query, candidates, top_k=self.n_final)
        else:
            candidates = candidates[: self.n_final]

        return candidates

    def build_prompt(self, query: str, results: list[SearchResult]) -> str:
        """
        Assemble le prompt LLM avec les passages récupérés.

        Args:
            query:   Requête originale.
            results: Passages récupérés et re-classés.

        Returns:
            Prompt formaté prêt pour le LLM.
        """
        context_parts = []
        for i, result in enumerate(results, start=1):
            context_parts.append(f"[{i}] {result.chunk.text}")

        context = "\n\n".join(context_parts)
        return self.context_template.format(query=query, context=context)

    def query(self, query: str) -> dict:
        """
        Exécute le pipeline RAG complet.

        Args:
            query: Question en texte libre.

        Returns:
            Dict avec "prompt", "sources" et "query".
        """
        results = self.retrieve(query)
        prompt = self.build_prompt(query, results)

        return {
            "query": query,
            "prompt": prompt,
            "sources": [
                {
                    "text": r.chunk.text,
                    "doc_id": r.chunk.doc_id,
                    "score": r.score,
                    "rank": r.rank,
                }
                for r in results
            ],
        }
