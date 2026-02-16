# agentic_ai/rules_engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class PolicyLimits:
    max_cli_pct: float = 50.0
    max_cld_pct: float = 50.0
    min_credit_limit: float = 1000.0
    max_credit_limit: float = 2_000_000.0


class RulesEngine:
    """
    Deterministic rule enforcement layer.
    """

    def __init__(self, limits: PolicyLimits | None = None):
        self.limits = limits or PolicyLimits()

    def enforce(
        self,
        *,
        action: str,
        prev_limit: float,
        raw_magnitude_pct: float,
        context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Enforce business & regulatory rules.
        Returns corrected magnitude and final limit.
        """
        context = context or {}
        notes = []

        magnitude_pct = float(raw_magnitude_pct)

        if action == "CLI":
            if magnitude_pct > self.limits.max_cli_pct:
                notes.append(f"CLI capped at {self.limits.max_cli_pct}%")
                magnitude_pct = self.limits.max_cli_pct

            delta = prev_limit * magnitude_pct / 100.0
            new_limit = prev_limit + delta

        elif action == "CLD":
            if magnitude_pct > self.limits.max_cld_pct:
                notes.append(f"CLD capped at {self.limits.max_cld_pct}%")
                magnitude_pct = self.limits.max_cld_pct

            delta = prev_limit * magnitude_pct / 100.0
            new_limit = prev_limit - delta

        else:  # HOLD
            magnitude_pct = 0.0
            new_limit = prev_limit

        if new_limit < self.limits.min_credit_limit:
            notes.append("Applied minimum credit limit floor")
            new_limit = self.limits.min_credit_limit

        if new_limit > self.limits.max_credit_limit:
            notes.append("Applied maximum credit limit cap")
            new_limit = self.limits.max_credit_limit

        return {
            "action": action,
            "magnitude_pct": round(magnitude_pct, 2),
            "prev_limit": round(prev_limit, 2),
            "new_limit": round(new_limit, 2),
            "rule_notes": notes,
        }
