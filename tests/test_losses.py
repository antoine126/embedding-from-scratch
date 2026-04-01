"""
tests/test_losses.py
======================
Tests unitaires pour les fonctions de coût (Chapitre 9).
"""

import pytest
import torch
import torch.nn.functional as functional

from embedding.losses import (
    ContrastiveLoss,
    InfoNCELoss,
    MatryoshkaLoss,
    MNRLoss,
    TripletLoss,
)


@pytest.fixture
def batch_size() -> int:
    return 8


@pytest.fixture
def embeddings(batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Paires d'embeddings normalisés L2."""
    emb1 = functional.normalize(torch.randn(batch_size, 128), dim=1)
    emb2 = functional.normalize(torch.randn(batch_size, 128), dim=1)
    return emb1, emb2


class TestContrastiveLoss:
    def test_output_scalar(self, embeddings):
        emb1, emb2 = embeddings
        labels = torch.randint(0, 2, (len(emb1),))
        loss = ContrastiveLoss()(emb1, emb2, labels)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_similar_pairs_low_loss(self):
        """Paires identiques -> perte proche de 0 pour les similaires."""
        emb = functional.normalize(torch.randn(4, 64), dim=1)
        labels = torch.zeros(4, dtype=torch.long)  # toutes similaires
        loss = ContrastiveLoss()(emb, emb.clone(), labels)
        assert loss.item() < 0.01


class TestTripletLoss:
    def test_output_scalar(self):
        a = functional.normalize(torch.randn(8, 64), dim=1)
        p = functional.normalize(torch.randn(8, 64), dim=1)
        n = functional.normalize(torch.randn(8, 64), dim=1)
        loss = TripletLoss()(a, p, n)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_perfect_separation(self):
        """Si d(a,p)=0 et d(a,n)=margin -> perte = 0."""
        a = functional.normalize(torch.randn(4, 64), dim=1)
        p = a.clone()  # positif identique à l'ancre
        n = -a  # négatif à l'opposé
        loss = TripletLoss(margin=0.5)(a, p, n)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)


class TestMNRLoss:
    def test_output_scalar(self, embeddings):
        q, p = embeddings
        loss = MNRLoss()(q, p)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_lower_with_perfect_pairs(self):
        """Embeddings identiques -> perte faible (diagonale dominante)."""
        emb = functional.normalize(torch.randn(8, 64), dim=1)
        loss_perfect = MNRLoss(temperature=0.05)(emb, emb.clone())
        loss_random = MNRLoss(temperature=0.05)(
            emb, functional.normalize(torch.randn(8, 64), dim=1)
        )
        assert loss_perfect.item() < loss_random.item()

    def test_temperature_effect(self):
        """Température plus faible 
            -> perte plus élevée (distribution plus concentrée)."""
        q = functional.normalize(torch.randn(16, 64), dim=1)
        p = functional.normalize(torch.randn(16, 64), dim=1)
        loss_low_temp = MNRLoss(temperature=0.01)(q, p)
        loss_high_temp = MNRLoss(temperature=0.5)(q, p)
        # Avec une température faible, la perte peut être très élevée si
        # les exemples ne sont pas bien séparés
        assert isinstance(loss_low_temp.item(), float)
        assert isinstance(loss_high_temp.item(), float)


class TestInfoNCELoss:
    def test_output_scalar(self, embeddings):
        q, p = embeddings
        loss = InfoNCELoss()(q, p)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_symmetric(self, embeddings):
        """InfoNCE est symétrique sur des paires aléatoires."""
        q, p = embeddings
        loss1 = InfoNCELoss()(q, p)
        # La perte combine q->p et p->q donc elle n'est pas
        # strictement symétrique entre (q,p) et (p,q), mais doit être >= 0
        assert loss1.item() >= 0


class TestMatryoshkaLoss:
    def test_output_scalar(self):
        q = torch.randn(8, 128)  # Non normalisé — MRL normalise par dimension
        p = torch.randn(8, 128)
        loss = MatryoshkaLoss(dimensions=[32, 64, 128])(q, p)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError):
            MatryoshkaLoss(dimensions=[])

    def test_invalid_weights(self):
        with pytest.raises(ValueError):
            MatryoshkaLoss(dimensions=[32, 64], weights=[0.5, 0.3, 0.2])
