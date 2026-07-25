r"""
metrics.py — privacy-aware model-evaluation metrics.

Each metric returns not just a value but its SENSITIVITY: how much one
individual's record can change the aggregate. Sensitivity is what lets the
release engine calibrate DP noise correctly -- get it wrong and the guarantee
is void, so it is computed explicitly per metric rather than assumed.

Metrics span the two things AI audits actually report:
  FAIRNESS  — does the model behave equitably across protected groups?
  SAFETY    — does the model refuse / avoid harmful outputs at acceptable rates?

All are group-wise rates (means of per-example indicators), so for a group of
size n the sensitivity of the RATE is 1/n (one flipped example moves a mean of
n bits by 1/n). We expose that explicitly.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class MetricResult:
    name: str
    group: str
    value: float
    sensitivity: float
    n: int


def _rate(indicator: np.ndarray, name: str, group: str) -> MetricResult:
    n = len(indicator)
    if n == 0:
        return MetricResult(name, group, 0.0, 1.0, 0)
    return MetricResult(name, group, float(indicator.mean()), 1.0 / n, n)


# ----------------------------------------------------------------- FAIRNESS
def selection_rate(y_pred, group_mask, group):
    """P(positive prediction | group). Basis of demographic parity."""
    return _rate(y_pred[group_mask].astype(float), "selection_rate", group)


def true_positive_rate(y_true, y_pred, group_mask, group):
    """P(pred=1 | true=1, group). Basis of equal-opportunity."""
    m = group_mask & (y_true == 1)
    if m.sum() == 0:
        return MetricResult("true_positive_rate", group, 0.0, 1.0, 0)
    return _rate((y_pred[m] == 1).astype(float), "true_positive_rate", group)


def false_positive_rate(y_true, y_pred, group_mask, group):
    m = group_mask & (y_true == 0)
    if m.sum() == 0:
        return MetricResult("false_positive_rate", group, 0.0, 1.0, 0)
    return _rate((y_pred[m] == 1).astype(float), "false_positive_rate", group)


# ----------------------------------------------------------------- SAFETY
def refusal_rate(refused, group_mask, group):
    """P(model refused | group) — safety coverage across groups."""
    return _rate(refused[group_mask].astype(float), "refusal_rate", group)


def harmful_output_rate(harmful, group_mask, group):
    """P(harmful output | group) — the rate you must prove is low & equitable."""
    return _rate(harmful[group_mask].astype(float), "harmful_output_rate", group)


# ----------------------------------------------------------------- DISPARITY
def disparity(results: list[MetricResult]) -> dict:
    """Max gap between groups for a metric — the headline fairness number.
    Its sensitivity is the max of the two groups' sensitivities (a change to
    one group moves at most one endpoint of the gap)."""
    if len(results) < 2:
        return {}
    vals = {r.group: r.value for r in results}
    hi = max(vals, key=vals.get); lo = min(vals, key=vals.get)
    sens = max(r.sensitivity for r in results)
    return {"metric": results[0].name, "max_group": hi, "min_group": lo,
            "gap": round(vals[hi] - vals[lo], 4), "sensitivity": sens}


FAIRNESS_METRICS = {"selection_rate": selection_rate,
                    "true_positive_rate": true_positive_rate,
                    "false_positive_rate": false_positive_rate}
SAFETY_METRICS = {"refusal_rate": refusal_rate,
                  "harmful_output_rate": harmful_output_rate}
