r"""
demo.py — end-to-end: audit a (deliberately biased) model on sensitive data
and produce a DP-certified report. Shows the whole wedge working.
"""
import numpy as np
from dp_model_audit.dp_release import PrivacySpec
from dp_model_audit.audit_report import DPAudit


def synthetic_eval(n=4000, seed=1):
    """A model with a PLANTED fairness problem and a small subgroup (to trigger
    k-floor suppression), plus safety labels."""
    rng = np.random.default_rng(seed)
    # groups with realistic imbalance; group D is tiny (will be suppressed)
    group = rng.choice(["A", "B", "C", "D"], size=n, p=[0.5, 0.3, 0.19, 0.01])
    y_true = rng.integers(0, 2, n)
    # model is fair-ish for A/B, biased against C (lower TPR), tiny D
    y_pred = y_true.copy()
    flip = rng.random(n)
    y_pred[(group == "C") & (y_true == 1) & (flip < 0.35)] = 0     # miss positives for C
    y_pred[(group == "A") & (y_true == 0) & (flip < 0.05)] = 1
    # safety: harmful outputs slightly elevated for one group
    harmful = (rng.random(n) < np.where(group == "C", 0.06, 0.02)).astype(int)
    refused = (rng.random(n) < 0.15).astype(int)
    return y_true, y_pred, group, harmful, refused


if __name__ == "__main__":
    y_true, y_pred, group, harmful, refused = synthetic_eval()
    print(f"eval set: {len(group)} examples across groups "
          f"{dict(zip(*np.unique(group, return_counts=True)))}\n")

    audit = DPAudit(PrivacySpec(epsilon=1.0, delta=1e-6, k_floor=50), seed=7)

    print("running fairness audit (equal-opportunity via TPR/FPR)...")
    audit.measure_fairness(y_true, y_pred, group,
                           ["true_positive_rate", "false_positive_rate"])
    print("running safety audit (harmful-output + refusal rates)...")
    audit.measure_safety(harmful, refused, group)

    report = audit.finalize()
    print("\n" + audit.to_markdown(report))

    # show the disparity computed from the RELEASED (noisy) numbers
    import metrics as M
    tpr = [M.true_positive_rate(y_true, y_pred, group == g, g)
           for g in ("A", "B", "C")]
    d = M.disparity(tpr)
    print(f"\n[true disparity, for reference] TPR gap {d['max_group']} vs "
          f"{d['min_group']} = {d['gap']} (the audit surfaces this while "
          f"protecting individuals)")
