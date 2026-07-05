"""HTV-Agent: hypothesis--test answer verification for VeriEvol.

Offline, refutation-first verifier that accepts a candidate answer only after
multi-source counter-evidence (multiple independent solvers, a skeptical
verifier with programmatic + visual tool channels, and a deterministic
acceptance gate) has failed to refute it. See the paper Section
"HTV-Agent: Hypothesis--Test Answer Verification".
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
