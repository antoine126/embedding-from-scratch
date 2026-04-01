"""Fonctions de coût contrastives (Chapitre 9)."""
from embedding.losses.contrastive import ContrastiveLoss
from embedding.losses.matryoshka import MatryoshkaLoss
from embedding.losses.mnr import InfoNCELoss, MNRLoss
from embedding.losses.triplet import TripletLoss

__all__ = [
    "ContrastiveLoss",
    "TripletLoss",
    "MNRLoss",
    "InfoNCELoss",
    "MatryoshkaLoss",
]
