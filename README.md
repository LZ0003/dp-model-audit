# dp-model-audit

**Differential-privacy-certified fairness & safety audits for ML models on sensitive evaluation data.**

Prove your model's fairness and safety metrics — to a regulator, an auditor, or the public — *without* the underlying records being reconstructable from the numbers you release.

Most AI-governance tools (model cards, bias dashboards) can't run on sensitive eval data with a guarantee. This is the missing layer: it computes group-wise fairness and safety metrics, releases them under `(ε, δ)`-differential privacy, tracks a privacy budget across the whole audit, suppresses groups below a minimum size, and emits a tamper-evident certified report.

## Install
```bash
pip install dp-model-audit
```

## Quickstart
```python
from dp_model_audit import DPAudit, PrivacySpec

audit = DPAudit(PrivacySpec(epsilon=1.0, delta=1e-6, k_floor=50))
audit.measure_fairness(y_true, y_pred, groups, ["true_positive_rate", "false_positive_rate"])
audit.measure_safety(harmful, refused, groups)

report = audit.finalize()          # dict; raises BudgetExceeded if over budget
print(audit.to_markdown(report))
```

## What you get
- **Every metric is DP-noised** — bounded worst-case leak about any individual.
- **A privacy budget ledger** — total exposure across *all* queries is bounded, not just each one. Run too many queries and it refuses, by design.
- **A k-floor** — no group smaller than `k` is ever reported (defeats singling-out before noise even applies).
- **A tamper-evident audit log** — every query, cost, result, hashed into the certificate.

## Why differential privacy (not just k-anonymity)
k-anonymity alone is defeated by combining multiple query results. DP tracks a mathematical *privacy budget* so an adversary can't average away the noise across many audits. The budget is the guarantee.

## Honest limitations (v0.1)
- Basic composition (conservative). A Rényi-DP accountant is on the roadmap for tighter budgets.
- Budget is split evenly across planned metrics; a budget *planner* (allocate ε to hit target confidence intervals) is coming.
- Small groups get large noise — this is correct DP behavior, and the report surfaces it rather than hiding it.

## Roadmap
Rényi-DP accountant · budget planner · signed certificates · connectors for scikit-learn, HuggingFace `evaluate`, and governance platforms.

## License
Apache-2.0 — inspect it, integrate it, ship it. Privacy tools you can't read aren't trustworthy.
