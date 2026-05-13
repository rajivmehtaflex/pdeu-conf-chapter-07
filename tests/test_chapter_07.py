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


def test_agent_builder_configures_interrupt(monkeypatch):
    calls = {}

    def fake_create_deep_agent(**kwargs):
        calls.update(kwargs)
        return "agent"

    monkeypatch.setattr(audit_agent, "create_deep_agent", fake_create_deep_agent)
    audit_agent.build_agent("test-model")

    assert calls["interrupt_on"] == {"request_cfo_approval": True}
    assert audit_agent.request_cfo_approval in calls["tools"]
