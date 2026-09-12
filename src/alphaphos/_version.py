"""Single source of truth for the package version.

Read by hatchling at build time (``[tool.hatch.version]`` in
``pyproject.toml``) and re-exported as ``alphaphos.__version__``.  Submodules
that stamp provenance import from here rather than from ``alphaphos`` so
they don't depend on package ``__init__`` import order.
"""

__version__ = "0.22.0"
