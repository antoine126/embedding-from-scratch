"""
scripts/demo_rag.py
====================
Chapitres 16-20 — Démonstration complète du pipeline RAG.

Ce script illustre le pipeline RAG de bout en bout :
  1. Découpage de documents en chunks
  2. Indexation avec le retriever hybride (dense + BM25)
  3. Requête avec recherche hybride + re-ranking
  4. Assemblage du prompt LLM

Usage :
    uv run python scripts/demo_rag.py

Note : Ce script utilise un modèle HuggingFace pré-entraîné (E5-base)
       pour la démo. Remplacez par votre modèle entraîné avec finetune.py.
"""
from __future__ import annotations

import torch
import torch.nn.functional as functional
from loguru import logger
from transformers import AutoModel, AutoTokenizer

from embedding.rag.chunker import Chunk, HierarchicalChunker
from embedding.rag.pipeline import HybridRetriever, RAGPipeline
from embedding.rag.retriever import BM25Retriever, DenseRetriever


class HuggingFaceEmbeddingModel(torch.nn.Module):
    """
    Wrapper pour utiliser un modèle HuggingFace comme retriever dense.
    Utilise le mean pooling standard sur les token embeddings.
    """

    def __init__(self, model_name: str = "intfloat/e5-small-v2") -> None:
        super().__init__()
        self.hf_model = AutoModel.from_pretrained(model_name)
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.hf_model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        # Mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        emb = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return functional.normalize(emb, p=2, dim=-1)


# Corpus de démonstration (extraits fictifs sur les embeddings)
DEMO_CORPUS = [
    {
        "id": "doc_1",
        "title": "Transformer et embeddings",
        "text": (
            "Un modèle d'embedding Transformer transforme une séquence de tokens "
            "en un vecteur dense dans un espace métrique. L'architecture encodeur "
            "utilise l'attention multi-têtes bidirectionnelle pour capturer les "
            "dépendances à longue distance dans le texte. Le pooling final agrège "
            "la séquence en un vecteur unique représentant le sens global."
        ),
    },
    {
        "id": "doc_2",
        "title": "MNR Loss et apprentissage contrastif",
        "text": (
            "La Multiple Negative Ranking Loss (MNR) est la fonction de coût "
            "standard pour l'entraînement des modèles d'embedding modernes. "
            "Elle utilise les autres exemples du batch comme négatifs in-batch, "
            "sans nécessiter d'annoter des exemples négatifs explicitement. "
            "Un grand batch améliore la qualité car il génère plus de négatifs."
        ),
    },
    {
        "id": "doc_3",
        "title": "Retrieval Augmented Generation",
        "text": (
            "Le RAG (Retrieval Augmented Generation) combine un système de "
            "récupération d'information avec un modèle de langage génératif. "
            "Les documents pertinents sont d'abord retrouvés par similarité "
            "sémantique, puis fournis comme contexte au LLM pour générer "
            "une réponse fondée sur des sources vérifiables."
        ),
    },
    {
        "id": "doc_4",
        "title": "Recherche hybride BM25 + dense",
        "text": (
            "La recherche hybride combine la recherche dense (embeddings) avec "
            "la recherche lexicale BM25. BM25 excelle pour les termes exacts "
            "et les entités nommées, tandis que la recherche dense gère mieux "
            "la synonymie et la paraphrase. La fusion par Reciprocal Rank Fusion "
            "combine les deux rankings sans normalisation des scores."
        ),
    },
    {
        "id": "doc_5",
        "title": "FAISS et indexation vectorielle",
        "text": (
            "FAISS (Facebook AI Similarity Search) est une bibliothèque pour "
            "la recherche approximative de plus proches voisins (ANN). "
            "IndexFlatIP implémente la recherche exacte par produit scalaire, "
            "équivalente à la similarité cosinus pour des vecteurs normalisés L2. "
            "IndexIVFPQ offre un compromis vitesse/précision pour les grands corpus."
        ),
    },
]


def main() -> None:
    logger.info("Démarrage de la démo RAG...")

    # --- Modèle d'embedding (HuggingFace E5-small pour la démo) ---
    logger.info("Chargement du modèle E5-small...")
    hf_model = HuggingFaceEmbeddingModel("intfloat/e5-small-v2")
    hf_model.eval()

    # --- Découpage des documents ---
    chunker = HierarchicalChunker(child_chunk_words=100, parent_chunk_words=300)
    all_chunks: list[Chunk] = []
    for doc in DEMO_CORPUS:
        chunks = chunker.chunk(doc["text"], doc_id=doc["id"], title=doc["title"])
        all_chunks.extend(chunks)
    logger.info(f"{len(all_chunks)} chunks créés depuis {len(DEMO_CORPUS)} documents")

    # --- Retriever dense ---
    dense_retriever = DenseRetriever(
        model=hf_model,
        tokenizer=hf_model.hf_tokenizer,
        device="cpu",
        batch_size=32,
    )

    # --- Retriever BM25 ---
    bm25_retriever = BM25Retriever()

    # --- Pipeline hybride ---
    hybrid = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        rrf_k=60,
        dense_weight=0.6,
    )
    pipeline = RAGPipeline(
        retriever=hybrid,
        n_retrieval=10,
        n_final=3,
    )

    # --- Indexation ---
    logger.info("Indexation des documents...")
    pipeline.index_documents(all_chunks)
    logger.success("Indexation terminée.")

    # --- Requêtes de démonstration ---
    queries = [
        "Comment fonctionne l'attention multi-têtes ?",
        "Qu'est-ce que la MNR Loss ?",
        "Comment combiner BM25 et la recherche dense ?",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"Requête : {query}")
        print("=" * 60)

        result = pipeline.query(query)

        for source in result["sources"]:
            print(f"\n[Rang {source['rank']}] Score: {source['score']:.4f}")
            print(f"Document: {source['doc_id']}")
            print(f"Extrait: {source['text'][:200]}...")

        print(f"\nPrompt LLM :\n{result['prompt'][:400]}...")


if __name__ == "__main__":
    main()
