# agentic_ai/decision_agent.py
from __future__ import annotations

from typing import Dict, Any

from agentic_ai.rules_engine import RulesEngine, PolicyLimits
from agentic_ai.rationale_generator import RationaleGenerator


class DecisionAgent:
    """
    Agentic decision-maker that combines
    model outputs + rules + rationale.
    """

    def __init__(
        self,
        *,
        limits: PolicyLimits | None = None,
    ):
        self.rules = RulesEngine(limits)
        self.rationale = RationaleGenerator()

    def decide(
        self,
        *,
        gate_action: str,
        magnitude_pct: float,
        prev_credit_limit: float,
        signals: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        signals = signals or {}

        # Step 1: enforce rules
        ruled = self.rules.enforce(
            action=gate_action,
            prev_limit=prev_credit_limit,
            raw_magnitude_pct=magnitude_pct,
            context=signals,
        )

        # Step 2: generate rationale
        explanation = self.rationale.generate(
            action=ruled["action"],
            magnitude_pct=ruled["magnitude_pct"],
            signals=signals,
            rule_notes=ruled["rule_notes"],
        )

        # Step 3: final agent output
        return {
            "action_taken": ruled["action"],
            "magnitude_percentage": ruled["magnitude_pct"],
            "prev_credit_limit": ruled["prev_limit"],
            "updated_credit_limit": ruled["new_limit"],
            "explanation": explanation,
        }
