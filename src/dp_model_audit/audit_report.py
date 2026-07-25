r"""
audit_report.py — the top-level object a governance platform integrates.

Ties metrics + DP release into one workflow that yields a certified,
human-reviewable audit report: the numbers, their noise bounds, the privacy
cost of producing them, and the certificate a regulator or counsel signs.

Usage (the whole integration surface):

    audit = DPAudit(PrivacySpec(epsilon=1.0, delta=1e-6, k_floor=50))
    audit.measure_fairness(y_true, y_pred, groups, ["true_positive_rate"])
    audit.measure_safety(harmful, refused, groups)
    report = audit.finalize()          # dict; also .to_markdown()
"""
import json
import hashlib
from dataclasses import asdict
import numpy as np

from .dp_release import DPReleaseEngine, PrivacySpec, BudgetExceeded, GroupTooSmall
from . import metrics as M


class DPAudit:
    def __init__(self, spec: PrivacySpec, per_query_epsilon: float = None, seed=0):
        self.spec = spec
        self.engine = DPReleaseEngine(spec, seed=seed)
        # split budget evenly if not told otherwise; caller can override per call
        self.default_eps = per_query_epsilon
        self.released = []
        self.suppressed = []

    def _eps_for(self, n_planned):
        if self.default_eps: return self.default_eps
        return self.spec.epsilon / max(1, n_planned)

    def _run(self, metric_fn, args_per_group, name, eps):
        out = []
        for group, args in args_per_group.items():
            res = metric_fn(*args, group)
            try:
                rel = self.engine.release_metric(
                    name, group, res.value, res.sensitivity, res.n, eps)
                out.append(rel)
                self.released.append(rel)
            except GroupTooSmall as e:
                self.suppressed.append({"group": group, "metric": name,
                                        "reason": str(e)})
            except BudgetExceeded as e:
                self.suppressed.append({"group": group, "metric": name,
                                        "reason": str(e)})
        return out

    def measure_fairness(self, y_true, y_pred, group_labels, which, eps=None):
        groups = np.unique(group_labels)
        eps = eps or self._eps_for(len(which) * len(groups))
        results = {}
        for metric_name in which:
            fn = M.FAIRNESS_METRICS[metric_name]
            args_per_group = {}
            for g in groups:
                mask = group_labels == g
                if metric_name == "selection_rate":
                    args_per_group[g] = (y_pred, mask)
                else:
                    args_per_group[g] = (y_true, y_pred, mask)
            results[metric_name] = self._run(fn, args_per_group, metric_name, eps)
        return results

    def measure_safety(self, harmful, refused, group_labels, eps=None):
        groups = np.unique(group_labels)
        eps = eps or self._eps_for(2 * len(groups))
        out = {}
        for name, arr in (("harmful_output_rate", harmful), ("refusal_rate", refused)):
            fn = M.SAFETY_METRICS[name]
            args_per_group = {g: (arr, group_labels == g) for g in groups}
            out[name] = self._run(fn, args_per_group, name, eps)
        return out

    def finalize(self) -> dict:
        summ = self.engine.summary()
        report = {
            "certificate": {
                "guarantee": f"({summ['epsilon_spent']}, {summ['delta']})-differential privacy",
                "interpretation": "Worst-case information any single individual's "
                                  "record contributes to this report is bounded by "
                                  f"epsilon={summ['epsilon_spent']}.",
                "k_floor": summ["k_floor"],
                "budget_status": "WITHIN BUDGET" if summ["epsilon_remaining"] >= 0 else "EXCEEDED",
            },
            "privacy_ledger": summ,
            "released_metrics": self.released,
            "suppressed": self.suppressed,
            "audit_log": [asdict(e) for e in self.engine.ledger.entries],
        }
        # tamper-evident hash of the full ledger
        blob = json.dumps(report["audit_log"], sort_keys=True, default=str)
        report["certificate"]["ledger_hash"] = hashlib.sha256(blob.encode()).hexdigest()[:16]
        return report

    def to_markdown(self, report=None) -> str:
        r = report or self.finalize()
        c = r["certificate"]; L = r["privacy_ledger"]
        lines = ["# Model Audit Report (DP-Certified)", "",
                 f"**Guarantee:** {c['guarantee']}",
                 f"**Budget:** {L['epsilon_spent']} / {L['epsilon_budget']} spent "
                 f"({L['epsilon_remaining']} remaining) · delta={L['delta']}",
                 f"**Min group size (k):** {c['k_floor']} · "
                 f"**Status:** {c['budget_status']}",
                 f"**Ledger hash:** `{c['ledger_hash']}`", "",
                 "## Released metrics (noisy, DP-protected)", "",
                 "| metric | group | value | ±95% CI | n |",
                 "|---|---|---|---|---|"]
        for m in r["released_metrics"]:
            lines.append(f"| {m['query']} | {m['group']} | {m['value']} "
                         f"| ±{m['ci95']} | {m['n']} |")
        if r["suppressed"]:
            lines += ["", "## Suppressed (privacy-protected)", ""]
            for s in r["suppressed"]:
                lines.append(f"- **{s['metric']} / {s['group']}**: {s['reason']}")
        lines += ["", "## Interpretation", "", c["interpretation"],
                  "", "_Every number above is noised to guarantee no individual's "
                  "data is reconstructable. The ledger hash makes the audit "
                  "trail tamper-evident._"]
        return "\n".join(lines)
