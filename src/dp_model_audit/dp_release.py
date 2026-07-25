r"""
dp_release.py — the certified-release engine.

The heart of the wedge: turn a raw aggregate metric into a differentially-
private released value, track the privacy budget it costs, enforce a minimum
group size, and refuse anything that would exceed the budget.

Privacy model: (epsilon, delta)-differential privacy via the Gaussian
mechanism, with budget composition tracked in a ledger. Each released
statistic has a SENSITIVITY (how much one person can change it); noise is
scaled to sensitivity/epsilon so the guarantee holds.
"""
from dataclasses import dataclass, field
from typing import Optional
import math
import numpy as np


class BudgetExceeded(Exception):
    pass


class GroupTooSmall(Exception):
    pass


@dataclass
class PrivacySpec:
    epsilon: float          # total privacy budget for the whole audit
    delta: float            # failure probability (typically 1e-6 .. 1e-9)
    k_floor: int = 50       # no group smaller than this is ever reported


@dataclass
class LedgerEntry:
    query: str
    group: str
    epsilon_spent: float
    sensitivity: float
    noisy_value: float
    raw_n: int
    timestamp: float


class PrivacyLedger:
    """Tracks cumulative privacy spend. Uses basic (advanced-composition-ready)
    accounting: epsilon adds across queries. A production version swaps in the
    Renyi-DP accountant; the interface is identical."""
    def __init__(self, spec: PrivacySpec):
        self.spec = spec
        self.entries: list[LedgerEntry] = []

    @property
    def spent(self) -> float:
        return sum(e.epsilon_spent for e in self.entries)

    @property
    def remaining(self) -> float:
        return self.spec.epsilon - self.spent

    def check(self, eps: float):
        if eps > self.remaining + 1e-12:
            raise BudgetExceeded(
                f"query needs eps={eps:.3f} but only {self.remaining:.3f} "
                f"of budget {self.spec.epsilon} remains")

    def record(self, entry: LedgerEntry):
        self.entries.append(entry)


class GaussianMechanism:
    """Adds calibrated Gaussian noise for (epsilon, delta)-DP.
    sigma = sensitivity * sqrt(2 ln(1.25/delta)) / epsilon."""
    def __init__(self, delta: float):
        self.delta = delta

    def sigma(self, sensitivity: float, epsilon: float) -> float:
        return sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / epsilon

    def release(self, value: float, sensitivity: float, epsilon: float,
                rng: np.random.Generator) -> tuple[float, float]:
        s = self.sigma(sensitivity, epsilon)
        return value + rng.normal(0, s), s


class DPReleaseEngine:
    """Coordinates: budget check -> k-floor check -> noise -> ledger record."""
    def __init__(self, spec: PrivacySpec, seed: int = 0):
        self.spec = spec
        self.ledger = PrivacyLedger(spec)
        self.mech = GaussianMechanism(spec.delta)
        self.rng = np.random.default_rng(seed)
        import time
        self._clock = time.time

    def release_metric(self, query: str, group: str, raw_value: float,
                       sensitivity: float, group_n: int,
                       epsilon: float) -> dict:
        """Release one group-wise metric under DP. Raises if the group is too
        small or the budget is exhausted."""
        if group_n < self.spec.k_floor:
            raise GroupTooSmall(
                f"group '{group}' has n={group_n} < k_floor={self.spec.k_floor}; "
                f"suppressed to prevent singling-out")
        self.ledger.check(epsilon)
        noisy, sigma = self.mech.release(raw_value, sensitivity, epsilon, self.rng)
        entry = LedgerEntry(query, group, epsilon, sensitivity, noisy,
                            group_n, self._clock())
        self.ledger.record(entry)
        # 95% confidence half-width from the noise (what the report shows)
        ci = 1.96 * sigma
        return {"query": query, "group": group, "value": round(noisy, 4),
                "ci95": round(ci, 4), "epsilon": epsilon, "n": group_n,
                "raw_suppressed": True}

    def summary(self) -> dict:
        return {"epsilon_budget": self.spec.epsilon,
                "epsilon_spent": round(self.ledger.spent, 4),
                "epsilon_remaining": round(self.ledger.remaining, 4),
                "delta": self.spec.delta, "k_floor": self.spec.k_floor,
                "queries": len(self.ledger.entries)}
