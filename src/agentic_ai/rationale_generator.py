# agentic_ai/rationale_generator.py
from __future__ import annotations

from typing import Dict, Any, List


class RationaleGenerator:
    """
    Generates human-readable rationale for decisions.
    """

    def generate(
        self,
        *,
        action: str,
        magnitude_pct: float,
        signals: Dict[str, Any],
        rule_notes: List[str] | None = None,
    ) -> str:
        rule_notes = rule_notes or []

        reasons = []

        if action == "HOLD":
            reasons.append(
                "your recent account indicators appeared stable with no strong signals for change"
            )

        elif action == "CLI":
            reasons.append(
                "your recent credit utilization, repayment behavior, and income stability showed improvement"
            )
            reasons.append(f"we increased your credit limit by approximately {magnitude_pct:.1f}%")

        elif action == "CLD":
            reasons.append(
                "there were recent signs of elevated risk such as higher utilization or repayment volatility"
            )
            reasons.append(f"we reduced your credit limit by approximately {magnitude_pct:.1f}%")

        # Optional signals (safe, non-sensitive)
        if signals.get("util_trend") == "up":
            reasons.append("your credit usage has been trending upward recently")
        if signals.get("payment_risk") == "high":
            reasons.append("we observed irregularities in recent payment patterns")

        explanation = "We reviewed your account and "

        explanation += ", and ".join(reasons)
        explanation += "."

        if rule_notes:
            explanation += " Note: " + "; ".join(rule_notes) + "."

        return explanation
