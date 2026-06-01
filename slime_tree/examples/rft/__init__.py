"""
RFT (Rejection Sampling Fine-Tuning) Example

Generates multiple candidates per prompt, filters by reward model,
and trains only on correct responses.
"""

from .rft_rollout import generate_rollout

__all__ = ["generate_rollout"]
