"""Pipeline RAG et recherche hybride (Chapitres 16-20)."""
from embedding.rag.chunker import Chunk, HierarchicalChunker, NaiveChunker, SentenceChunker
from embedding.rag.pipeline import HybridRetriever, RAGPipeline
from embedding.rag.reranker import CrossEncoderReranker
from embedding.rag.retriever import BM25Retriever, DenseRetriever, SearchResult

__all__ = [
    "Chunk",
    "NaiveChunker",
    "SentenceChunker",
    "HierarchicalChunker",
    "DenseRetriever",
    "BM25Retriever",
    "SearchResult",
    "CrossEncoderReranker",
    "HybridRetriever",
    "RAGPipeline",
]
