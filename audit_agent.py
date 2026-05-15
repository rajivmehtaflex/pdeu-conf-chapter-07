from __future__ import annotations

import argparse
import json
import os
import sqlite3
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv
from deepagents import create_deep_agent
from loguru import logger

ROOT = Path(__file__).resolve().parent
ENV_PATH = ROOT / ".env"
DEFAULT_MODEL = "openrouter:inclusionai/ring-2.6-1t:free"
from pydantic import BaseModel, Field

CONTRACTS_DIR = ROOT / "contracts"
DB_PATH = ROOT / "ap_ledger.db"
DELIVERY_LOG_PATH = ROOT / "warehouse_receipts_fy26.csv"
LEDGER_SCHEMA_HINT = (
    "Tables: Vendors, Invoices, Payments. Columns: Vendors(Vendor_ID, Vendor_Name, GSTIN, Address), "
    "Invoices(Invoice_ID, Vendor_ID, Amount, Invoice_Date, Status), "
    "Payments(Payment_ID, Invoice_ID, Amount, Payment_Date). "
    "Payments uses Amount for the payment amount; there is no Payment_Amount column."
)

SYSTEM_PROMPT = """
You are the Senior Financial Auditor for Shree Manufacturing Pvt. Ltd.
You MUST use write_todos to outline a 3-step audit plan before taking any other action.
Use query_ledger for accounts payable data, check_delivery_log for warehouse receipts, and read_file for legal contracts.
Calculate late-delivery penalties from the contract clause and report any discrepancy greater than INR 0.
""".strip()


def _connect():
    import sqlite3
    return sqlite3.connect(DB_PATH)


def query_ledger(sql: str) -> str:
    """Execute a read-only SQL query against the accounts payable ledger."""
    lowered = sql.strip().lower()
    if not lowered.startswith("select"):
        raise ValueError("Only SELECT statements are allowed")
    with _connect() as con:
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute(sql).fetchall()
        except sqlite3.Error as exc:
            return json.dumps(
                {
                    "error": "Invalid SELECT for this ledger schema",
                    "message": str(exc),
                    "schema_hint": LEDGER_SCHEMA_HINT,
                }
            )
    return json.dumps([dict(row) for row in rows])


def check_delivery_log(vendor_id: str) -> str:
    """Return warehouse receipt rows for a vendor without exposing unrelated rows."""
    import pandas as pd
    frame = pd.read_csv(DELIVERY_LOG_PATH)
    rows = frame.loc[frame["Vendor_ID"] == vendor_id].copy()
    rows["days_late"] = (
        pd.to_datetime(rows["Actual_Delivery"]) - pd.to_datetime(rows["Expected_Delivery"])
    ).dt.days
    return rows.to_json(orient="records")


def find_contract(vendor_name: str) -> Path:
    path = CONTRACTS_DIR / (vendor_name.replace(" ", "_") + "_Contract.txt")
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_contract(vendor_name: str) -> str:
    return find_contract(vendor_name).read_text(encoding="utf-8")


def read_file(vendor_name: str) -> str:
    """Read a contract file for a vendor by name."""
    try:
        return read_contract(vendor_name)
    except FileNotFoundError:
        return f"Contract not found for vendor: {vendor_name}. Available contracts are in {CONTRACTS_DIR}"


def get_vendor_id(vendor_name: str) -> str:
    rows = json.loads(query_ledger(f"select Vendor_ID from Vendors where Vendor_Name = '{vendor_name}'"))
    if not rows:
        raise ValueError(f"Unknown vendor: {vendor_name}")
    return rows[0]["Vendor_ID"]


def build_discrepancy_summary(vendor_name: str) -> dict[str, Any]:
    vendor_id = get_vendor_id(vendor_name)
    invoices = json.loads(query_ledger(f"select Invoice_ID, Vendor_ID, Amount, Status from Invoices where Vendor_ID = '{vendor_id}'"))
    deliveries = json.loads(check_delivery_log(vendor_id))
    max_late = max(row["days_late"] for row in deliveries)
    invoice_amount = float(invoices[0]["Amount"])
    penalty = invoice_amount * 0.05 if max_late > 7 and "5% penalty" in read_contract(vendor_name) else 0.0
    return {
        "vendor_id": vendor_id,
        "vendor_name": vendor_name,
        "invoice_amount": invoice_amount,
        "days_late": int(max_late),
        "penalty_amount_inr": penalty,
        "action_required": "Recover Funds" if penalty else "None",
    }


class DiscrepancyReport(BaseModel):
    vendor_id: str = Field(description="The unique ID of the vendor")
    vendor_name: str
    invoice_amount: float
    days_late: int
    penalty_amount_inr: float
    action_required: str = Field(description="e.g., 'Recover Funds', 'None'")


def build_discrepancy_report(vendor_name: str) -> DiscrepancyReport:
    return DiscrepancyReport(**build_discrepancy_summary(vendor_name))


def report_to_json(report: DiscrepancyReport) -> str:
    return report.model_dump_json(indent=2)


def requires_cfo_approval(penalty_amount: float) -> bool:
    return penalty_amount > 10000


def request_cfo_approval(vendor_name: str, penalty_amount: float) -> str:
    """Request CFO approval before finalizing a high-value penalty report."""
    return f"CFO approval required for {vendor_name}: INR {penalty_amount:,.2f}"


def prompt_for_cfo_approval(
    *,
    vendor_name: str,
    penalty_amount: float,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[[str], None] = print,
) -> bool:
    output_fn(
        "Agent is requesting CFO Approval for "
        f"an INR {penalty_amount:,.2f} penalty on {vendor_name}."
    )
    answer = input_fn("Approve? (y/n) ").strip().lower()
    return answer in {"y", "yes"}


def finalize_report(vendor_name: str, approved: bool = True) -> DiscrepancyReport:
    report = build_discrepancy_report(vendor_name)
    if requires_cfo_approval(report.penalty_amount_inr) and not approved:
        return report.model_copy(update={"action_required": "None - Overridden by CFO"})
    return report


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _known_vendor_names() -> list[str]:
    rows = json.loads(query_ledger("select Vendor_Name from Vendors"))
    return sorted((row["Vendor_Name"] for row in rows), key=len, reverse=True)


def _find_vendor_in_prompt(prompt: str) -> str | None:
    normalized_prompt = _normalize(prompt)
    for vendor_name in _known_vendor_names():
        if _normalize(vendor_name) in normalized_prompt:
            return vendor_name
    return None


def build_augmented_prompt(prompt: str) -> str:
    vendor_name = _find_vendor_in_prompt(prompt)
    if vendor_name is None:
        return prompt

    vendor_id = get_vendor_id(vendor_name)
    invoices = query_ledger(
        "select Invoice_ID, Vendor_ID, Amount, Invoice_Date, Status "
        f"from Invoices where Vendor_ID = '{vendor_id}'"
    )
    deliveries = check_delivery_log(vendor_id)
    contract_text = read_contract(vendor_name)
    return (
        f"{prompt}\n\n"
        "LOCAL REFERENCE DATA\n"
        f"Vendor: {vendor_name}\n"
        f"Vendor ID: {vendor_id}\n"
        f"DB Schema Hint: {LEDGER_SCHEMA_HINT} "
        "Use Vendors.Vendor_ID to join Vendors and Invoices, then Payments.Invoice_ID to join Payments; there is no accounts_payable table.\n"
        "Tool Guidance: query_ledger accepts read-only SELECT SQL. "
        f"Use check_delivery_log(\"{vendor_id}\") for warehouse receipts and read_file(\"{vendor_name}\") for the contract.\n"
        "Response Shape: return a DiscrepancyReport with vendor_id, vendor_name, invoice_amount, "
        "days_late, penalty_amount_inr, and action_required.\n"
        "Approval Guidance: if penalty_amount_inr > INR 10000, call request_cfo_approval before the final report.\n\n"
        "Known-good SQL examples:\n"
        f"- select Vendor_ID, Vendor_Name from Vendors where Vendor_Name = '{vendor_name}'\n"
        f"- select Invoice_ID, Vendor_ID, Amount, Invoice_Date, Status from Invoices where Vendor_ID = '{vendor_id}'\n"
        f"- select Payment_ID, Invoice_ID, Amount, Payment_Date from Payments where Invoice_ID = 'INV-2000'\n\n"
        "Contract:\n"
        f"{contract_text}\n\n"
        "Invoice Rows:\n"
        f"{invoices}\n\n"
        "Delivery Rows:\n"
        f"{deliveries}\n"
    )


def build_agent(model_name: str):
    return create_deep_agent(
        model=model_name,
        tools=[query_ledger, check_delivery_log, read_file, request_cfo_approval],
        system_prompt=SYSTEM_PROMPT + "\nIf the calculated penalty is > INR 10000, you MUST call request_cfo_approval before generating the final DiscrepancyReport.",
        response_format=DiscrepancyReport,
        interrupt_on={"request_cfo_approval": True},
    )


def run_self_check() -> str:
    return report_to_json(finalize_report("Gujarat Steel Corp", approved=True))


def run_approval_demo(vendor_name: str = "Gujarat Steel Corp") -> str:
    report = build_discrepancy_report(vendor_name)
    approved = True
    if requires_cfo_approval(report.penalty_amount_inr):
        approved = prompt_for_cfo_approval(
            vendor_name=vendor_name,
            penalty_amount=report.penalty_amount_inr,
        )
    return report_to_json(finalize_report(vendor_name, approved=approved))


def load_model_name() -> str:
    load_dotenv(ENV_PATH)
    return os.getenv("OPENROUTER_MODEL") or os.getenv("MODEL_NAME") or DEFAULT_MODEL


def invoke_agent(prompt: str) -> str:
    agent = build_agent(load_model_name())
    result = agent.invoke({"messages": [{"role": "user", "content": build_augmented_prompt(prompt)}]})
    if result.get("__interrupt__"):
        return "CFO approval required before finalizing the DiscrepancyReport."
    messages = result.get("messages", [])
    if not messages:
        return ""
    final = messages[-1]
    return str(getattr(final, "content", final))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", default="What is your job?")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--approval-demo", action="store_true")
    args = parser.parse_args(argv)

    if args.self_check:
        print(run_self_check())
        return 0

    if args.approval_demo:
        print(run_approval_demo())
        return 0

    print(invoke_agent(args.prompt))
    return 0
