# DP Model-Audit Wedge — design notes & what the demo proved

## What it is
A runnable reference implementation of the ONE capability AI-governance
platforms lack: DP-certified release of fairness/safety metrics computed on
sensitive evaluation data. ~400 lines, three clean modules, one integration
object (`DPAudit`).

## What the demo proved (audit of a deliberately biased model, 4000 examples)
- Surfaced a planted fairness gap: group C true-positive-rate 0.61 vs ~1.0 for
  A/B (a real 0.36 disparity) WHILE noising every released number.
- k-floor suppression fired: group D (n=37) never reported — defeats
  singling-out before DP even applies.
- Budget ledger fired: after spending eps=1.0 on fairness metrics, safety
  queries were correctly REFUSED ("0.000 budget remains"). You cannot average
  away the noise by running more queries — the guarantee holds across the whole
  audit, not just per query.
- Tamper-evident: full audit log hashed into the certificate.

## Two honest design lessons from the run
1. **Budget must be pre-allocated, not first-come.** The demo spent its whole
   budget on fairness and starved safety. A real integration reserves eps per
   planned metric up front. (Interface already supports per-call eps; the
   orchestrator should plan the split.)
2. **Noise vs group size is a real, surfaced tension.** Group C's FPR came out
   negative — DP noise swamping a modest-n rate. This is CORRECT behavior and
   the tool exposes it rather than hiding it: auditing many metrics on modest
   groups needs more budget or fewer metrics. That honesty is the selling
   point to a regulator.

## Why this is the right WEDGE (not a platform)
- Governance platforms (model cards, bias dashboards, policy engines) are
  crowded and mostly CANNOT run on sensitive eval data with a guarantee.
- This supplies exactly that missing layer, behind a one-object interface that
  drops into any evaluation pipeline: `DPAudit(spec).measure_*().finalize()`.
- The moat is the math + the auditable ledger, not UI. Integrate INTO a
  governance stack; become the default privacy layer it calls.

## Production roadmap (what a real version adds)
- Renyi-DP / zCDP accountant for tighter composition (interface is identical;
  swap `PrivacyLedger`). Basic composition here is conservative on purpose.
- Budget PLANNER: given a metric plan, allocate eps to hit target CIs, or warn
  that the plan is infeasible at the chosen budget.
- More metrics: calibration, subgroup AUC, counterfactual fairness.
- Signed certificates (real cryptographic signature, not just a hash) for
  regulator hand-off.
- Connectors: scikit-learn / HF `evaluate` / a governance platform's eval API.

## Integration surface (the whole thing)
```python
audit = DPAudit(PrivacySpec(epsilon=1.0, delta=1e-6, k_floor=50))
audit.measure_fairness(y_true, y_pred, groups, ["true_positive_rate"])
audit.measure_safety(harmful, refused, groups)
report = audit.finalize()          # dict + .to_markdown(); raises if over budget
```
One object, one guarantee, drops into any model-evaluation pipeline.
