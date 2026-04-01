"""
tests/test_model.py
=====================
Tests unitaires pour l'architecture du modèle (Chapitres 5-8).
"""
import pytest
import torch

from embedding.model.attention import MultiHeadAttention
from embedding.model.config import ModelConfig
from embedding.model.embeddings import (
    PositionalEncoding,
    TokenEmbedding,
)
from embedding.model.encoder import EmbeddingModel
from embedding.model.layers import FeedForward, Pooling


@pytest.fixture
def small_config() -> ModelConfig:
    """Configuration minimale pour les tests rapides."""
    return ModelConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        max_seq_len=32,
        vocab_size=1000,
        dropout=0.0,
        pooling="mean",
    )


@pytest.fixture
def batch(small_config: ModelConfig) -> dict[str, torch.Tensor]:
    """Batch synthétique : 4 séquences de 16 tokens."""
    B, L = 4, 16
    input_ids = torch.randint(1, small_config.vocab_size, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)
    # Simuler du padding sur les 4 derniers tokens de chaque séquence
    attention_mask[:, -4:] = 0
    return {"input_ids": input_ids, "attention_mask": attention_mask}


class TestTokenEmbedding:
    def test_output_shape(self, small_config, batch):
        emb = TokenEmbedding(small_config)
        out = emb(batch["input_ids"])
        assert out.shape == (4, 16, 64)

    def test_padding_zero(self, small_config):
        """Le token [PAD] (id=0) doit donner un vecteur nul."""
        emb = TokenEmbedding(small_config)
        pad_ids = torch.zeros(1, 1, dtype=torch.long)
        out = emb(pad_ids)
        assert out.abs().sum().item() == 0.0


class TestPositionalEncoding:
    @pytest.mark.parametrize("kind", ["sinusoidal", "learned"])
    def test_output_shape(self, kind, small_config, batch):
        small_config.pos_encoding = kind
        pe = PositionalEncoding(small_config)
        x = torch.randn(4, 16, 64)
        out = pe(x)
        assert out.shape == (4, 16, 64)


class TestMultiHeadAttention:
    def test_output_shape(self, small_config, batch):
        attn = MultiHeadAttention(small_config)
        x = torch.randn(4, 16, 64)
        out = attn(x, batch["attention_mask"])
        assert out.shape == (4, 16, 64)

    def test_d_model_divisibility_check(self):
        with pytest.raises(AssertionError):
            config = ModelConfig(d_model=64, n_heads=5)  # 64 % 5 != 0
            MultiHeadAttention(config)


class TestFeedForward:
    @pytest.mark.parametrize("activation", ["gelu", "relu", "swiglu", "geglu"])
    def test_output_shape(self, activation, small_config):
        small_config.activation = activation
        ffn = FeedForward(small_config)
        x = torch.randn(4, 16, 64)
        out = ffn(x)
        assert out.shape == (4, 16, 64)


class TestPooling:
    @pytest.mark.parametrize("strategy", ["cls", "mean", "weighted_mean"])
    def test_output_shape(self, strategy, small_config, batch):
        small_config.pooling = strategy
        pooling = Pooling(small_config)
        hidden = torch.randn(4, 16, 64)
        out = pooling(hidden, batch["attention_mask"])
        assert out.shape == (4, 64)

    def test_mean_pooling_ignores_padding(self, small_config):
        """Les tokens de padding ne doivent pas contribuer au mean pooling."""
        pooling = Pooling(small_config)
        hidden = torch.randn(2, 8, 64)
        mask = torch.ones(2, 8, dtype=torch.long)
        mask[0, 4:] = 0  # Padding sur la deuxième moitié de la première séquence

        out_masked = pooling(hidden, mask)

        # Recalcul manuel
        expected = hidden[0, :4].mean(0)
        assert torch.allclose(out_masked[0], expected, atol=1e-5)


class TestEmbeddingModel:
    def test_output_shape(self, small_config, batch):
        model = EmbeddingModel(small_config)
        out = model(batch["input_ids"], batch["attention_mask"])
        assert out.shape == (4, 64)

    def test_l2_normalized(self, small_config, batch):
        """Les sorties doivent être normalisées L2."""
        model = EmbeddingModel(small_config)
        out = model(batch["input_ids"], batch["attention_mask"])
        norms = out.norm(dim=-1)
        assert torch.allclose(norms, torch.ones(4), atol=1e-5)

    def test_config_validation(self):
        with pytest.raises(ValueError, match="divisible"):
            ModelConfig(d_model=64, n_heads=5)
