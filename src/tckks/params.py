"""CKKS parameter sets evaluated in the report.

Every set targets 128-bit classical security (HE Standard) and lets OpenFHE pick
the smallest ring dimension that satisfies it. The sets differ in how much
multiplicative depth fits in one "epoch" before a collective refresh is needed.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ParamSet:
    name: str
    mult_depth: int          # levels available per epoch
    scale_bits: int          # log2 of the CKKS scaling factor Delta
    first_mod_bits: int      # size of q0
    num_large_digits: int    # dnum for HYBRID key switching
    description: str
    rot_strategy: str = "all"  # "all": keys for shifts 1..7; "pow2": keys for 1, 2, 4 only

    @property
    def epoch_len(self) -> int:
        """GD iterations per epoch.

        OpenFHE's interactive bootstrapping (IntMPBootAdjustScale) needs at least
        two unused levels in its input, so two levels are held in reserve. Part 2
        budgeted one; testing showed one is not enough (see report, Section 5).
        """
        return max(1, self.mult_depth - 2)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["epoch_len"] = self.epoch_len
        return d


PARAM_SETS = {
    # Small ring, shallow circuit: cheapest per operation, refresh every 3 iterations.
    "P1-compact": ParamSet("P1-compact", mult_depth=5, scale_bits=40, first_mod_bits=60,
                           num_large_digits=3, description="N=2^14, refresh every 3 iterations"),
    # The Part 2 design point: N=2^15, L=10, Delta=2^50, refresh every 8 iterations.
    "P2-design": ParamSet("P2-design", mult_depth=10, scale_bits=50, first_mod_bits=60,
                          num_large_digits=3, description="Part 2 design: N=2^15, L=10, refresh every 8"),
    # Deep circuit: all 26 products of K=27 in one epoch, no refresh, larger ring.
    "P3-deep": ParamSet("P3-deep", mult_depth=28, scale_bits=45, first_mod_bits=60,
                        num_large_digits=4, description="N=2^16, no refresh: all 26 products at depth 28",
                        rot_strategy="pow2"),
}

# Public protocol constants (Part 2, Section 2)
D = 8                 # features including intercept
K_DEFAULT = 27        # gradient-descent iterations
N_BOUND = 2 ** 16     # public bound on total records
MIN_CELL = 11         # small-cell suppression threshold
ROUND_DIGITS = 4      # significant digits released
