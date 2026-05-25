"""Readers for MaxQuant, Spectronaut, DIA-NN, AlphaPept outputs into PhosphoExperiment."""

from alphaphos.io.spectronaut import read_psm as read_spectronaut

__all__ = ["read_spectronaut"]
