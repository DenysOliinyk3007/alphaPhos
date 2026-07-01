"""Differential testing.

Currently exposes a two-group moderated t-test via limma (inmoose backend).
ANOVA / multi-contrast F-tests are intentionally out of scope for 0.5.0.
"""

from alphaphos.stats.diff_exp import DEFAULT_STATS_SETTINGS, diff_exp_limma

__all__ = ["diff_exp_limma", "DEFAULT_STATS_SETTINGS"]
