import json
import audit_agent


def test_approval_required_for_large_penalty():
    assert audit_agent.requires_cfo_approval(25000) is True
    assert audit_agent.requires_cfo_approval(10000) is False


def test_finalize_report_honors_cfo_override():
    report = audit_agent.finalize_report("Gujarat Steel Corp", approved=False)
    assert report.penalty_amount_inr == 25000.0
    assert report.action_required == "None - Overridden by CFO"


def test_prompt_for_cfo_approval_reads_user_decision():
    seen = []

    approved = audit_agent.prompt_for_cfo_approval(
        vendor_name="Gujarat Steel Corp",
        penalty_amount=25000,
        input_fn=lambda _: "n",
        output_fn=seen.append,
    )

    assert approved is False
    assert "CFO Approval" in seen[0]
    assert "Gujarat Steel Corp" in seen[0]


def test_query_ledger_returns_schema_guidance_for_malformed_select():
    result = json.loads(audit_agent.query_ledger("select p.Payment_Amount from Payments p"))

    assert result["error"] == "Invalid SELECT for this ledger schema"
    assert "Tables: Vendors, Invoices, Payments" in result["schema_hint"]


def test_agent_builder_configures_interrupt(monkeypatch):
    calls = {}

    def fake_create_deep_agent(**kwargs):
        calls.update(kwargs)
        return "agent"

    monkeypatch.setattr(audit_agent, "create_deep_agent", fake_create_deep_agent)
    audit_agent.build_agent("test-model")

    assert calls["interrupt_on"] == {"request_cfo_approval": True}
    assert audit_agent.request_cfo_approval in calls["tools"]


def test_invoke_agent_reports_cfo_interrupt(monkeypatch):
    class EmptyMessage:
        content = ""

    class FakeAgent:
        def invoke(self, _payload):
            return {"messages": [EmptyMessage()], "__interrupt__": ("request_cfo_approval",)}

    monkeypatch.setattr(audit_agent, "load_model_name", lambda: "test-model")
    monkeypatch.setattr(audit_agent, "build_agent", lambda _model: FakeAgent())

    result = audit_agent.invoke_agent("Audit the account for Gujarat Steel Corp.")

    assert "CFO approval required" in result


def test_build_augmented_prompt_includes_cfo_threshold_for_known_vendor():
    prompt = audit_agent.build_augmented_prompt("Audit the account for Gujarat Steel Corp.")

    assert "Gujarat Steel Corp" in prompt
    assert "VEN-1000" in prompt
    assert "DiscrepancyReport" in prompt
    assert "penalty_amount_inr" in prompt
    assert "request_cfo_approval" in prompt
    assert "> INR 10000" in prompt
