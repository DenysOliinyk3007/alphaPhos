"""Bundled data resources for alphaPhos.

Currently contains the MaxQuant-derived ``contaminants.fasta`` used by
``alphaphos.preprocess.contaminants.filter_contaminants`` as the default
contaminant database. Access via :func:`importlib.resources.files`:

    >>> from importlib.resources import files
    >>> files("alphaphos.resources") / "contaminants.fasta"
"""
