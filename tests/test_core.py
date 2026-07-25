import numpy as np
from dp_model_audit import DPAudit, PrivacySpec, BudgetExceeded

def test_budget_enforced():
    a = DPAudit(PrivacySpec(epsilon=0.1, delta=1e-6, k_floor=10), per_query_epsilon=0.08)
    g = np.array(["A"]*100 + ["B"]*100)
    yt = np.random.default_rng(0).integers(0,2,200); yp = yt.copy()
    a.measure_fairness(yt, yp, g, ["true_positive_rate"])
    # second call should exhaust budget -> suppressed, not crash
    a.measure_fairness(yt, yp, g, ["false_positive_rate"])
    assert any("budget" in s["reason"].lower() for s in a.suppressed)

def test_k_floor():
    a = DPAudit(PrivacySpec(epsilon=1.0, delta=1e-6, k_floor=50))
    g = np.array(["big"]*100 + ["tiny"]*10)
    yt = np.zeros(110,int); yp = np.zeros(110,int)
    a.measure_fairness(yt, yp, g, ["selection_rate"])
    assert any(s["group"]=="tiny" for s in a.suppressed)

def test_report_has_certificate():
    a = DPAudit(PrivacySpec(epsilon=1.0, delta=1e-6, k_floor=10))
    g = np.array(["A"]*60+["B"]*60); yt=np.random.default_rng(1).integers(0,2,120)
    a.measure_fairness(yt, yt.copy(), g, ["selection_rate"])
    r = a.finalize()
    assert "ledger_hash" in r["certificate"]
    assert r["certificate"]["budget_status"] == "WITHIN BUDGET"
