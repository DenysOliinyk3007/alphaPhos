"""Differential testing.

* :func:`diff_exp_limma` -- two-group moderated t-test (inmoose backend).
* :func:`diff_exp_limma_contrasts` -- multi-contrast moderated t-tests
  on one joint fit (alphaPhos own moderation stack, or an inmoose loop
  fallback via ``joint=False``).
* :func:`diff_exp_anova` -- moderated F-test across all condition
  levels (ANOVA-style "does anything differ?").
* :func:`design_matrix` -- typed design-matrix builder used by the above.

The moderated multi-contrast + ANOVA path is a clean-room implementation
of Smyth 2004 (see :mod:`alphaphos.stats.moderated` and
:mod:`alphaphos.stats.linear_model`) and does not depend on inmoose.
"""

from alphaphos.stats.design import DesignMatrix, design_matrix
from alphaphos.stats.diff_exp import (
    DEFAULT_STATS_SETTINGS,
    anova_hits,
    diff_exp_anova,
    diff_exp_limma,
    diff_exp_limma_contrasts,
)

__all__ = [
    "diff_exp_limma",
    "diff_exp_limma_contrasts",
    "diff_exp_anova",
    "anova_hits",
    "design_matrix",
    "DesignMatrix",
    "DEFAULT_STATS_SETTINGS",
]
