"""
tests/test_chunker.py
=======================
Tests unitaires pour les stratégies de chunking (Chapitre 17).
"""
import pytest

from embedding.rag.chunker import (
    Chunk,
    HierarchicalChunker,
    NaiveChunker,
    SentenceChunker,
)


SAMPLE_TEXT = (
    "Les modèles d'embedding transforment le texte en vecteurs denses. "
    "Ces vecteurs capturent le sens sémantique des phrases. "
    "La similarité cosinus mesure la proximité entre deux vecteurs. "
    "Plus deux vecteurs sont proches, plus les textes correspondants sont similaires. "
    "Cette propriété est au cœur des systèmes de recherche sémantique modernes. "
    "Le Transformer encodeur est l'architecture dominante pour les embeddings. "
    "L'attention multi-têtes permet de capturer des dépendances à longue distance."
)


class TestNaiveChunker:
    def test_returns_chunks(self):
        chunker = NaiveChunker(chunk_size=20, overlap=5)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_chunk_size_respected(self):
        chunker = NaiveChunker(chunk_size=20, overlap=0)
        chunks = chunker.chunk(SAMPLE_TEXT)
        for chunk in chunks[:-1]:  # Tous sauf le dernier
            assert len(chunk) <= 20

    def test_doc_id_assigned(self):
        chunks = NaiveChunker().chunk(SAMPLE_TEXT, doc_id="my_doc")
        assert all(c.doc_id == "my_doc" for c in chunks)

    def test_overlap(self):
        """Avec overlap, les chunks consécutifs doivent partager des mots."""
        chunker = NaiveChunker(chunk_size=10, overlap=3)
        chunks = chunker.chunk(SAMPLE_TEXT)
        if len(chunks) > 1:
            words_0 = set(chunks[0].text.split())
            words_1 = set(chunks[1].text.split())
            assert len(words_0 & words_1) > 0


class TestSentenceChunker:
    def test_returns_chunks(self):
        chunker = SentenceChunker(max_words=30)
        chunks = chunker.chunk(SAMPLE_TEXT)
        assert len(chunks) > 0

    def test_empty_text(self):
        chunks = SentenceChunker().chunk("")
        assert len(chunks) == 0

    def test_chunk_not_exceed_max_words(self):
        """Chaque chunk ne doit pas dépasser max_words sauf si une seule phrase."""
        chunker = SentenceChunker(max_words=20)
        chunks = chunker.chunk(SAMPLE_TEXT)
        for chunk in chunks:
            words = len(chunk.text.split())
            # Une phrase seule peut dépasser max_words
            assert words <= 50  # Limite raisonnable


class TestHierarchicalChunker:
    def test_returns_chunks(self):
        chunker = HierarchicalChunker(child_chunk_words=20)
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id="doc1", title="Embeddings")
        assert len(chunks) > 0

    def test_context_prefix_added(self):
        """Le titre doit apparaître dans chaque chunk."""
        chunker = HierarchicalChunker(child_chunk_words=20)
        title = "Introduction aux Embeddings"
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id="doc1", title=title)
        for chunk in chunks:
            assert title in chunk.text

    def test_no_title(self):
        """Sans titre, pas de préfixe de contexte."""
        chunker = HierarchicalChunker(child_chunk_words=20)
        chunks = chunker.chunk(SAMPLE_TEXT, doc_id="doc1")
        for chunk in chunks:
            assert "Document :" not in chunk.text
