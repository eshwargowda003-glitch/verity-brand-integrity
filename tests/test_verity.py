import os
import tempfile
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.main import (
    app,
    Base,
    db_dep,
    init_db,
    compile_phrase_pattern,
    run_engine,
    calc_status,
    Rule,
    Finding,
)

@pytest.fixture(scope="function")
def test_client():
    # Create isolated temporary database file
    temp_dir = tempfile.mkdtemp()
    temp_db_path = os.path.join(temp_dir, "test_verity.db")
    test_engine = create_engine(
        f"sqlite:///{temp_db_path}",
        connect_args={"check_same_thread": False}
    )
    TestingSessionLocal = sessionmaker(test_engine, autoflush=False, expire_on_commit=False)
    init_db(test_engine)

    def override_db_dep():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[db_dep] = override_db_dep
    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()
        test_engine.dispose()
        try:
            if os.path.exists(temp_db_path):
                os.remove(temp_db_path)
            os.rmdir(temp_dir)
        except Exception:
            pass


# ==============================================================================
# TASK 1 TESTS: RULE MATCHING & ENGINE TESTS
# ==============================================================================

def test_rule_matching_exact():
    rule = Rule(
        id=1, brand_id=1, kind="prohibited_phrase", title="Prohibit Free",
        pattern="free", severity="high", explanation="No free claims",
        suggestion="complimentary", source_note="Policy A"
    )
    findings = run_engine("This is free shipping for you.", [rule])
    assert len(findings) == 1
    assert findings[0]["matched_text"] == "free"
    assert findings[0]["evidence_status"] == "rule_matched"
    assert findings[0]["evidence_status"] != "contradicted"


def test_rule_matching_case_insensitivity():
    rule = Rule(
        id=1, brand_id=1, kind="prohibited_phrase", title="Prohibit Free",
        pattern="free", severity="high", explanation="No free claims",
        suggestion="complimentary", source_note="Policy A"
    )
    findings_upper = run_engine("GET FREE SHIPPING NOW!", [rule])
    assert len(findings_upper) == 1
    assert findings_upper[0]["matched_text"] == "FREE"

    findings_mixed = run_engine("Get FrEe delivery today.", [rule])
    assert len(findings_mixed) == 1
    assert findings_mixed[0]["matched_text"] == "FrEe"


def test_rule_matching_multiword_phrases():
    rule = Rule(
        id=2, brand_id=1, kind="prohibited_phrase", title="Prohibit Best In Class",
        pattern="best in class", severity="medium", explanation="Unsubstantiated superlative",
        suggestion="leading", source_note="Policy B"
    )
    # Single space
    f1 = run_engine("We are the best in class provider.", [rule])
    assert len(f1) == 1
    assert f1[0]["matched_text"] == "best in class"

    # Multiple whitespace and newlines
    f2 = run_engine("We are the best   in\nclass provider.", [rule])
    assert len(f2) == 1
    assert f2[0]["matched_text"] == "best   in\nclass"


def test_rule_matching_punctuation():
    rule = Rule(
        id=1, brand_id=1, kind="prohibited_phrase", title="Prohibit Free",
        pattern="free", severity="high", explanation="No free claims",
        suggestion="", source_note=""
    )
    # Preceded / followed by punctuation
    f1 = run_engine("Is it really 'free'?", [rule])
    assert len(f1) == 1
    assert f1[0]["matched_text"] == "free"

    f2 = run_engine("Completely free!", [rule])
    assert len(f2) == 1
    assert f2[0]["matched_text"] == "free"

    f3 = run_engine("(free)", [rule])
    assert len(f3) == 1
    assert f3[0]["matched_text"] == "free"


def test_rule_matching_partial_word_false_positives():
    rule = Rule(
        id=1, brand_id=1, kind="prohibited_phrase", title="Prohibit Free",
        pattern="free", severity="high", explanation="No free claims",
        suggestion="", source_note=""
    )
    # Must NOT flag 'freewheel' or 'notfree'
    assert len(run_engine("We bought a freewheel bicycle.", [rule])) == 0
    assert len(run_engine("This is notfree at all.", [rule])) == 0
    assert len(run_engine("Enjoy the carefree afternoon.", [rule])) == 0

    # MUST flag 'free shipping'
    findings = run_engine("Enjoy free shipping today.", [rule])
    assert len(findings) == 1
    assert findings[0]["matched_text"] == "free"


def test_rule_matching_empty_and_invalid_patterns():
    # Empty string or whitespace pattern
    assert compile_phrase_pattern("") is None
    assert compile_phrase_pattern("   ") is None
    rule_empty = Rule(id=1, brand_id=1, kind="prohibited_phrase", title="Empty", pattern="", severity="low", explanation="x")
    assert len(run_engine("Some sample text", [rule_empty])) == 0


def test_rule_matching_literal_regex_escaping():
    rule = Rule(
        id=3, brand_id=1, kind="prohibited_phrase", title="Special characters",
        pattern="100% (guaranteed)", severity="high", explanation="No guarantees",
        suggestion="", source_note=""
    )
    # Escaping preserves parentheses and percent symbol literally without regex error
    f = run_engine("We offer 100% (guaranteed) returns.", [rule])
    assert len(f) == 1
    assert f[0]["matched_text"] == "100% (guaranteed)"


# ==============================================================================
# TASK 2 TESTS: STATUS TRANSITIONS & DECISION LOGIC
# ==============================================================================

def test_calc_status_logic():
    # 0 findings -> ready_for_approval
    assert calc_status([], "") == "ready_for_approval"

    # Pending high / critical -> changes_required
    f_high = Finding(id=1, review_id=1, severity="high", decision_status="pending")
    f_low = Finding(id=2, review_id=1, severity="low", decision_status="pending")
    assert calc_status([f_high, f_low]) == "changes_required"

    # Only pending low/medium -> needs_review
    assert calc_status([f_low]) == "needs_review"

    # Accepted finding (active violation acknowledged) -> changes_required
    f_low_accepted = Finding(id=2, review_id=1, severity="low", decision_status="accepted")
    assert calc_status([f_low_accepted]) == "changes_required"

    # Mixed: 1 resolved, 1 accepted -> changes_required
    f_high_resolved = Finding(id=1, review_id=1, severity="high", decision_status="resolved")
    assert calc_status([f_high_resolved, f_low_accepted]) == "changes_required"

    # All resolved -> ready_for_approval (not automatically approved!)
    f_low_resolved = Finding(id=2, review_id=1, severity="low", decision_status="resolved")
    assert calc_status([f_high_resolved, f_low_resolved]) == "ready_for_approval"

    # All rejected -> ready_for_approval (not automatically approved!)
    f_high_rejected = Finding(id=1, review_id=1, severity="high", decision_status="rejected")
    f_low_rejected = Finding(id=2, review_id=1, severity="low", decision_status="rejected")
    assert calc_status([f_high_rejected, f_low_rejected]) == "ready_for_approval"


# ==============================================================================
# TASK 6 TESTS: END-TO-END WORKFLOW SUITE
# ==============================================================================

def test_root_endpoint(test_client):
    r = test_client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "VERITY Brand Integrity API"
    assert data["docs_url"] == "/docs"
    assert data["status"] == "online"


def test_1_health_endpoint(test_client):
    r = test_client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "database": "connected"}


def test_2_brand_creation(test_client):
    r = test_client.post("/api/brands", json={"name": "  Acme Corp  ", "description": "Global enterprise"})
    assert r.status_code == 201
    data = r.json()
    assert data["name"] == "Acme Corp"
    assert data["id"] > 0

    # Duplicate brand rejection
    r_dup = test_client.post("/api/brands", json={"name": "acme corp", "description": "Duplicate"})
    assert r_dup.status_code == 409


def test_3_and_4_rule_creation_and_retrieval(test_client):
    b = test_client.post("/api/brands", json={"name": "RuleBrand"}).json()
    bid = b["id"]

    # Rule creation with whitespace stripping and validation
    r = test_client.post(f"/api/brands/{bid}/rules", json={
        "kind": "prohibited_phrase",
        "title": "  No Free Claims  ",
        "pattern": "  free  ",
        "severity": "high",
        "explanation": "Free claims are strictly prohibited without disclosure",
        "suggestion": "complimentary",
        "source_note": "Governance Guideline 4.1"
    })
    assert r.status_code == 201
    rule_data = r.json()
    assert rule_data["pattern"] == "free"
    assert rule_data["title"] == "No Free Claims"

    # Duplicate rule rejection for same brand
    r_dup = test_client.post(f"/api/brands/{bid}/rules", json={
        "kind": "prohibited_phrase",
        "title": "Another title",
        "pattern": "FREE",
        "severity": "low",
        "explanation": "Duplicate pattern check"
    })
    assert r_dup.status_code == 409

    # Retrieve rules
    rules_res = test_client.get(f"/api/brands/{bid}/rules")
    assert rules_res.status_code == 200
    assert len(rules_res.json()) == 1


def test_5_and_6_review_creation_and_rule_matching(test_client):
    b = test_client.post("/api/brands", json={"name": "ReviewBrand"}).json()
    test_client.post(f"/api/brands/{b['id']}/rules", json={
        "kind": "prohibited_phrase",
        "title": "No Free",
        "pattern": "free",
        "severity": "high",
        "explanation": "No free terms allowed",
        "suggestion": "complimentary"
    })

    # Submit content with prohibited phrase
    r = test_client.post("/api/reviews", json={
        "brand_id": b["id"],
        "title": "Spring Campaign",
        "content": "Sign up now and get free shipping on your first purchase!"
    })
    assert r.status_code == 201
    rev = r.json()
    assert rev["status"] == "changes_required"  # High severity finding pending
    assert len(rev["findings"]) == 1
    assert rev["findings"][0]["matched_text"] == "free"
    assert rev["findings"][0]["decision_status"] == "pending"
    assert rev["findings"][0]["evidence_status"] == "rule_matched"
    assert rev["findings"][0]["evidence_status"] != "contradicted"


def test_7_incorrect_substring_matching(test_client):
    b = test_client.post("/api/brands", json={"name": "SubstringBrand"}).json()
    test_client.post(f"/api/brands/{b['id']}/rules", json={
        "kind": "prohibited_phrase",
        "title": "No Free",
        "pattern": "free",
        "severity": "high",
        "explanation": "No free terms allowed"
    })

    # Content with substrings 'freewheel' and 'notfree'
    r = test_client.post("/api/reviews", json={
        "brand_id": b["id"],
        "title": "Bicycle Promo",
        "content": "Check out our newest freewheel gear, it is notfree but affordable."
    })
    assert r.status_code == 201
    rev = r.json()
    # Should NOT trigger any findings!
    assert len(rev["findings"]) == 0
    assert rev["status"] == "ready_for_approval"


def test_8_finding_retrieval(test_client):
    b = test_client.post("/api/brands", json={"name": "RetrievalBrand"}).json()
    test_client.post(f"/api/brands/{b['id']}/rules", json={
        "kind": "prohibited_phrase",
        "title": "No Free",
        "pattern": "free",
        "severity": "medium",
        "explanation": "Explanation"
    })
    rev = test_client.post("/api/reviews", json={
        "brand_id": b["id"],
        "title": "Ad",
        "content": "Free items inside"
    }).json()

    get_rev = test_client.get(f"/api/reviews/{rev['id']}")
    assert get_rev.status_code == 200
    assert len(get_rev.json()["findings"]) == 1


def test_9_valid_decision_recording(test_client):
    b = test_client.post("/api/brands", json={"name": "DecisionBrand"}).json()
    test_client.post(f"/api/brands/{b['id']}/rules", json={
        "kind": "prohibited_phrase",
        "title": "No Free",
        "pattern": "free",
        "severity": "high",
        "explanation": "Explanation",
        "suggestion": "complimentary"
    })
    rev = test_client.post("/api/reviews", json={
        "brand_id": b["id"],
        "title": "Ad",
        "content": "Free sample"
    }).json()
    fid = rev["findings"][0]["id"]

    # Resolve finding
    r_dec = test_client.post(f"/api/reviews/{rev['id']}/decisions", json={
        "finding_id": fid,
        "decision": "resolved",
        "reason": "Applied suggested correction 'complimentary'"
    })
    assert r_dec.status_code == 200
    updated = r_dec.json()
    assert updated["findings"][0]["decision_status"] == "resolved"
    assert updated["findings"][0]["decision_reason"] == "Applied suggested correction 'complimentary'"
    assert updated["status"] == "ready_for_approval"


def test_10_invalid_decision_rejection(test_client):
    b = test_client.post("/api/brands", json={"name": "InvalidDecBrand"}).json()
    test_client.post(f"/api/brands/{b['id']}/rules", json={
        "kind": "prohibited_phrase",
        "title": "No Free",
        "pattern": "free",
        "severity": "low",
        "explanation": "Explanation"
    })
    rev = test_client.post("/api/reviews", json={
        "brand_id": b["id"],
        "content": "Free item"
    }).json()
    fid = rev["findings"][0]["id"]

    # Unsupported decision
    r_bad = test_client.post(f"/api/reviews/{rev['id']}/decisions", json={
        "finding_id": fid,
        "decision": "invalid_value"
    })
    assert r_bad.status_code == 422


def test_11_review_status_recalculation_and_approval(test_client):
    b = test_client.post("/api/brands", json={"name": "WorkflowBrand"}).json()
    test_client.post(f"/api/brands/{b['id']}/rules", json={
        "kind": "prohibited_phrase",
        "title": "No Free",
        "pattern": "free",
        "severity": "high",
        "explanation": "High severity rule",
        "suggestion": "complimentary"
    })
    test_client.post(f"/api/brands/{b['id']}/rules", json={
        "kind": "prohibited_phrase",
        "title": "No Quick",
        "pattern": "quick",
        "severity": "low",
        "explanation": "Low severity rule"
    })
    rev = test_client.post("/api/reviews", json={
        "brand_id": b["id"],
        "content": "Get free items with quick delivery"
    }).json()
    assert rev["status"] == "changes_required"
    findings = rev["findings"]
    f_high = next(f for f in findings if f["severity"] == "high")
    f_low = next(f for f in findings if f["severity"] == "low")

    # Cannot approve when status is changes_required
    appr_fail = test_client.post(f"/api/reviews/{rev['id']}/approve")
    assert appr_fail.status_code == 400

    # Resolve high finding
    r1 = test_client.post(f"/api/reviews/{rev['id']}/decisions", json={
        "finding_id": f_high["id"],
        "decision": "resolved"
    }).json()
    # Now only low finding is pending -> status becomes 'needs_review'
    assert r1["status"] == "needs_review"

    # Reject low finding as false positive
    r2 = test_client.post(f"/api/reviews/{rev['id']}/decisions", json={
        "finding_id": f_low["id"],
        "decision": "rejected",
        "reason": "Permitted context"
    }).json()
    # Both dispositioned -> status becomes 'ready_for_approval'
    assert r2["status"] == "ready_for_approval"

    # Explicit human approval
    appr_ok = test_client.post(f"/api/reviews/{rev['id']}/approve")
    assert appr_ok.status_code == 200
    assert appr_ok.json()["status"] == "approved"


def test_12_report_generation(test_client):
    b = test_client.post("/api/brands", json={"name": "ReportBrand"}).json()
    test_client.post(f"/api/brands/{b['id']}/rules", json={
        "kind": "prohibited_phrase",
        "title": "Prohibited Word",
        "pattern": "banned",
        "severity": "critical",
        "explanation": "Strictly disallowed word",
        "suggestion": "permitted term"
    })
    rev = test_client.post("/api/reviews", json={
        "brand_id": b["id"],
        "title": "Compliance Document",
        "content": "This contains banned text."
    }).json()

    rep_res = test_client.get(f"/api/reviews/{rev['id']}/report")
    assert rep_res.status_code == 200
    report = rep_res.json()

    assert report["report_type"] == "VERITY Brand Governance & Compliance Report"
    assert report["brand"]["name"] == "ReportBrand"
    assert report["review"]["id"] == rev["id"]
    assert report["review"]["submitted_content"] == "This contains banned text."
    assert "does not independently verify" in report["disclaimer"]
    assert len(report["findings"]) == 1
    assert report["findings"][0]["matched_text"] == "banned"
    assert report["findings"][0]["suggested_correction"] == "permitted term"
    assert report["findings"][0]["is_correction_grounded"] is True
    assert report["findings"][0]["evidence_status"] == "rule_matched"
    assert report["findings"][0]["evidence_status"] != "contradicted"
    assert report["findings"][0]["rule_match_status"] == "matched"
    assert report["findings"][0]["verification_status"] == "unverified_external"
    assert len(report["outstanding_unresolved_findings"]) == 1

    # Also verify passport backward-compatibility
    pass_res = test_client.get(f"/api/reviews/{rev['id']}/passport")
    assert pass_res.status_code == 200
    passport = pass_res.json()
    assert "review" in passport
    assert "report" in passport


def test_13_missing_resources_handling(test_client):
    assert test_client.get("/api/brands/99999/rules").status_code == 404
    assert test_client.post("/api/brands/99999/rules", json={
        "kind": "prohibited_phrase", "title": "Valid Title", "pattern": "P", "explanation": "Explanation"
    }).status_code == 404
    assert test_client.get("/api/reviews/99999").status_code == 404
    assert test_client.get("/api/reviews/99999/report").status_code == 404
    assert test_client.post("/api/reviews/99999/decisions", json={"finding_id": 1, "decision": "resolved"}).status_code == 404


def test_14_persistence_across_requests(test_client):
    b = test_client.post("/api/brands", json={"name": "PersistentBrand"}).json()
    test_client.post(f"/api/brands/{b['id']}/rules", json={
        "kind": "prohibited_phrase", "title": "Rule 1", "pattern": "stop", "explanation": "Rule 1 note"
    })
    rev = test_client.post("/api/reviews", json={
        "brand_id": b["id"], "title": "Check 1", "content": "Please stop now"
    }).json()

    # Re-fetch from fresh request
    reviews = test_client.get("/api/reviews").json()
    assert any(r["id"] == rev["id"] for r in reviews)


def test_evidence_status_semantics(test_client):
    # Tests that rule matching does NOT claim factual contradiction
    b = test_client.post("/api/brands", json={"name": "EvidenceBrand"}).json()
    
    # 1. Prohibited phrase rule
    test_client.post(f"/api/brands/{b['id']}/rules", json={
        "kind": "prohibited_phrase",
        "title": "No Free",
        "pattern": "free shipping",
        "severity": "high",
        "explanation": "Free claims require disclosure",
        "suggestion": "complimentary shipping"
    })
    
    # 2. Required term rule
    test_client.post(f"/api/brands/{b['id']}/rules", json={
        "kind": "required_term",
        "title": "Terms Apply",
        "pattern": "terms apply",
        "severity": "medium",
        "explanation": "Must state terms apply"
    })
    
    # 3. Fact check rule
    test_client.post(f"/api/brands/{b['id']}/rules", json={
        "kind": "fact_check",
        "title": "Founding Year",
        "pattern": "Est. 1995",
        "severity": "low",
        "explanation": "Approved founding year"
    })
    
    rev = test_client.post("/api/reviews", json={
        "brand_id": b["id"],
        "title": "Shipping Promo",
        "content": "Get free shipping on all orders."
    }).json()
    
    findings = rev["findings"]
    f_prohibited = next(f for f in findings if f["category"] == "prohibited_phrase")
    f_required = next(f for f in findings if f["category"] == "required_term")
    f_fact = next(f for f in findings if f["category"] == "fact_check")
    
    # Ensure prohibited phrase is labeled 'rule_matched', NEVER 'contradicted'
    assert f_prohibited["evidence_status"] == "rule_matched"
    assert f_prohibited["evidence_status"] != "contradicted"
    
    # Ensure required term omission is labeled 'unverified'
    assert f_required["evidence_status"] == "unverified"
    
    # Ensure missing fact check value is labeled 'unverified'
    assert f_fact["evidence_status"] == "unverified"
    
    # Check decision transitions on evidence status
    # Resolve prohibited finding
    res_dec = test_client.post(f"/api/reviews/{rev['id']}/decisions", json={
        "finding_id": f_prohibited["id"],
        "decision": "resolved"
    }).json()
    f_resolved = next(f for f in res_dec["findings"] if f["id"] == f_prohibited["id"])
    assert f_resolved["evidence_status"] == "verified_resolved"
    
    # Dismiss / Reject required term finding
    rej_dec = test_client.post(f"/api/reviews/{rev['id']}/decisions", json={
        "finding_id": f_required["id"],
        "decision": "rejected",
        "reason": "Not applicable on social banner"
    }).json()
    f_rejected = next(f for f in rej_dec["findings"] if f["id"] == f_required["id"])
    assert f_rejected["evidence_status"] == "dismissed"
    
    # Accept fact check violation
    acc_dec = test_client.post(f"/api/reviews/{rev['id']}/decisions", json={
        "finding_id": f_fact["id"],
        "decision": "accepted"
    }).json()
    f_accepted = next(f for f in acc_dec["findings"] if f["id"] == f_fact["id"])
    assert f_accepted["evidence_status"] == "confirmed_violation"
