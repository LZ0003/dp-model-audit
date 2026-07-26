# DP-Certified Model Audit — the AI-governance wedge

## What this is
A focused engine that answers one question no incumbent answers cleanly:

  "Prove your model's fairness/safety metrics on sensitive data,
   in a report you can hand a regulator, WITHOUT the underlying
   records being reconstructable from the numbers you release."

It is deliberately NOT a platform. It is the **certified-release core** that a
larger AI-governance product (bias auditing, model cards, compliance
reporting) plugs in when its evaluation data is sensitive. That is the wedge:
governance platforms measure models; almost none can do it on protected data
with a mathematical guarantee. This supplies exactly that missing layer.

## Why a wedge, not a platform
- Governance platforms (model cards, audit workflows, policy) are crowded.
- Distribution strategy: integrate into an existing governance stack rather
  than compete with it. The moat is the math + the audit trail, not the UI.

## The three components
1. `metrics.py`   -- privacy-aware evaluation: computes group-wise model
                     metrics (fairness + safety) as SENSITIVITY-BOUNDED
                     aggregates, so DP noise can be calibrated correctly.
2. `dp_release.py`-- the certified-release engine: adds calibrated noise,
                     tracks an (epsilon, delta) budget ledger, enforces
                     minimum group sizes (k-anonymity floor), and refuses
                     releases that would exceed budget.
3. `audit_report.py`-- produces the signed, human-reviewable audit artifact:
                     the numbers, their noise bounds, the privacy cost, and
                     the certificate a regulator/counsel signs off on.

## The guarantee stack (why a regulator trusts the output)
- Every released metric carries (epsilon, delta) DP: bounded worst-case
  information leak about any one individual.
- A privacy LEDGER composes cost across all queries in an audit -- so the
  TOTAL exposure of running many fairness checks is bounded, not just each one.
- A k-floor: no group smaller than k is ever reported (defeats singling-out
  even before DP noise).
- An immutable audit log: every query, its cost, its result, its reviewer.

## Integration surface (how a governance platform uses it)
```
governance_platform.evaluate(model, sensitive_eval_set)
   -> DPAudit(budget=(eps, delta), k_floor=50)
       .measure(fairness=[demographic_parity, equal_opportunity],
                safety=[refusal_rate, toxicity_rate], group_by="ethnicity")
       .release()            # returns certified report or raises BudgetExceeded
```
One object, one guarantee, drops into any evaluation pipeline.
