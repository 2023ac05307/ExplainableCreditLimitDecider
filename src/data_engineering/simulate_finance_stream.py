#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simulate_finance_stream_v2_rl_magnitude.py
-----------------------------------------
Advanced, industry-grade credit-card portfolio simulator for ML/RL/XAI research.

ROOT FIXES (for realistic utilization + CLI candidates):
A) Revolve payment bug fixed (revolve payments were being overwritten by min_due).
B) Spend is tied to credit-limit utilization target (already in your tx generator).
C) Candidate + action decision uses PRE-payment utilization (demand signal).
D) Rolling utilization window tracks util_pre (not util_post).

Dependencies: numpy, pandas
"""
from __future__ import annotations

import os
import json
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------
# Action space
# ---------------------------------
ACTION_HOLD = 0
ACTION_CLI = 1
ACTION_CLD = 2

ACTION_NAME = {
    ACTION_HOLD: "HOLD",
    ACTION_CLI: "CLI",
    ACTION_CLD: "CLD",
}

# ---------------------------------
# Configs
# ---------------------------------

@dataclass
class MacroConfig:
    regimes: Tuple[str, ...] = ("boom", "normal", "recession")
    transition: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.85, 0.13, 0.02],
                [0.10, 0.80, 0.10],
                [0.03, 0.20, 0.77],
            ],
            dtype=float,
        )
    )
    base_unemp: Dict[str, float] = None
    inflation_annual: Dict[str, float] = None

    def __post_init__(self):
        if self.base_unemp is None:
            self.base_unemp = {"boom": 0.03, "normal": 0.055, "recession": 0.09}
        if self.inflation_annual is None:
            self.inflation_annual = {"boom": 0.015, "normal": 0.035, "recession": 0.06}


@dataclass
class CustomerConfig:
    n_customers: int = 10000
    start_date: str = "2006-01-01"
    years: int = 20
    seed: int = 42
    # Base distributions
    income_log_mean: float = 10.5
    income_log_sigma: float = 0.55
    credit_limit_factor_low: float = 0.08
    credit_limit_factor_high: float = 0.35
    monthly_spend_prop_base: float = 0.22
    pay_full_prob_base: float = 0.55
    roll_rate_base: float = 0.12
    cure_rate_base: float = 0.55
    default_hazard_base: float = 0.007
    churn_hazard_base: float = 0.002
    monthly_multipliers: List[float] = None

    def __post_init__(self):
        if self.monthly_multipliers is None:
            self.monthly_multipliers = [
                0.92, 0.96, 0.98, 1.00, 1.03, 1.05,
                1.04, 1.02, 1.01, 1.10, 1.18, 1.24
            ]


@dataclass
class OutputConfig:
    out_dir: str = "sim_output_v2"
    write_csv: bool = True
    partition_by_year: bool = True
    realtime_sleep: float = 0.0
    avg_tx_per_cust_per_month: float = 18.0
    statement_day: int = 20
    apr_annual: float = 0.24
    late_fee: float = 29.0
    overlimit_fee: float = 35.0
    min_pay_rate: float = 0.04
    min_pay_amount_floor: float = 25.0

    # Magnitude bounds
    cli_pct_min: float = 0.05
    cli_pct_max: float = 0.40
    cld_pct_min: float = 0.05
    cld_pct_max: float = 0.40

    # Credit limit floors/ceils
    credit_limit_floor: float = 5000.0
    credit_limit_cap: float = 800000.0

    # Logging
    log_hold_events: bool = False


# ---------------------------------
# Utils
# ---------------------------------

def month_end(dt: datetime) -> datetime:
    nm = dt.replace(day=28) + timedelta(days=4)
    return nm - timedelta(days=nm.day)

def month_start(dt: datetime) -> datetime:
    return dt.replace(day=1)

def year_month_key(dt: datetime) -> str:
    return dt.strftime("%Y-%m")

def clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

def choice_from_probs(probs: np.ndarray) -> int:
    return int(np.searchsorted(np.cumsum(probs), np.random.rand()))


# ---------------------------------
# Simulator
# ---------------------------------

class FinanceSimulatorV2:
    def __init__(self, cconf: CustomerConfig, mconf: MacroConfig, oconf: OutputConfig):
        self.cc = cconf
        self.mc = mconf
        self.oc = oconf
        np.random.seed(self.cc.seed)

        self.start_date = datetime.fromisoformat(self.cc.start_date)
        self.end_date = self.start_date + timedelta(days=int(self.cc.years * 365.25))

        self.customers = self._init_customers()
        self.regime_idx = 1
        self.regime = self.mc.regimes[self.regime_idx]

        self.apr_daily = self.oc.apr_annual / 365.0

        # Buffers
        self.transactions: List[Dict] = []
        self.statements: List[Dict] = []
        self.credit_events: List[Dict] = []
        self.income_history: List[Dict] = []
        self.derived_monthly: List[Dict] = []

        # Rolling state per customer
        n = self.cc.n_customers
        self.tx_amt_ema = np.zeros(n, dtype=float)
        self.tx_cnt_ema = np.zeros(n, dtype=float)
        self.overlimit_ema = np.zeros(n, dtype=float)
        self.util_windows: Dict[int, deque] = {i + 1: deque(maxlen=6) for i in range(n)}
        self.last_cli_limit = np.array(self.customers["credit_limit"].values, dtype=float)
        self.dpd_count_12m = np.zeros(n, dtype=int)

    # ---------- Initialization ----------
    def _init_customers(self) -> pd.DataFrame:
        n = self.cc.n_customers

        income = np.random.lognormal(self.cc.income_log_mean, self.cc.income_log_sigma, size=n)
        monthly_income = income / 12.0

        factors = np.random.uniform(self.cc.credit_limit_factor_low, self.cc.credit_limit_factor_high, size=n)
        credit_limit = np.clip(factors * income, self.oc.credit_limit_floor, 200000.0)

        # ROOT FIX A support: heterogeneous revolve propensity (0..0.9)
        revolve_prop = np.clip(np.random.beta(2, 6, size=n), 0.0, 0.9)

        pay_full_prob = np.clip(np.random.normal(self.cc.pay_full_prob_base, 0.15, size=n), 0.05, 0.95)
        monthly_spend_prop = np.clip(np.random.normal(self.cc.monthly_spend_prop_base, 0.08, size=n), 0.05, 0.75)

        # --- Premium features ---
        external_score = np.clip(np.random.normal(720, 60, size=n), 300, 900)
        num_open_accounts = np.random.poisson(4, size=n) + 1
        num_closed_accounts = np.random.poisson(2, size=n)
        total_open_credit = np.clip(np.random.normal(1_000_000, 400_000, size=n), 50_000, 3_000_000)
        external_delinquency_12m = np.maximum(0, np.random.poisson(0.2, size=n) - (external_score > 760).astype(int))

        residential_stability_years = np.clip(np.random.normal(5.5, 3.0, size=n), 0, 30)
        employment_stability_years = np.clip(np.random.normal(4.0, 2.5, size=n), 0, 30)
        industry_risk_score = np.clip(np.random.normal(0.0, 1.0, size=n), -2.5, 2.5)

        df = pd.DataFrame(
            {
                "cust_id": np.arange(1, n + 1, dtype=np.int64),
                "monthly_income": monthly_income,
                "credit_limit": credit_limit,
                "pay_full_prob": pay_full_prob,
                "monthly_spend_prop": monthly_spend_prop,
                "balance": np.zeros(n),
                "status": np.array(["current"] * n, dtype=object),
                "utilization": np.zeros(n),
                "join_date": [self.start_date] * n,
                "last_stmt_date": [None] * n,
                "months_on_book": np.zeros(n, dtype=int),
                "hardship": np.zeros(n, dtype=int),
                "jobless": np.zeros(n, dtype=int),
                "cli_eligible": np.ones(n, dtype=int),
                # Bureau-like
                "external_score": external_score,
                "num_open_accounts": num_open_accounts,
                "num_closed_accounts": num_closed_accounts,
                "total_open_credit": total_open_credit,
                "external_delinquency_12m": external_delinquency_12m,
                # Stability
                "residential_stability_years": residential_stability_years,
                "employment_stability_years": employment_stability_years,
                "industry_risk_score": industry_risk_score,
                # ROOT FIX A input
                "revolve_prop": revolve_prop,
            }
        )
        return df

    # ---------- Macro ----------
    def _advance_macro(self):
        P = self.mc.transition
        self.regime_idx = choice_from_probs(P[self.regime_idx])
        self.regime = self.mc.regimes[self.regime_idx]

    def _macro_multipliers(self) -> Dict[str, float]:
        if self.regime == "boom":
            return {"spend": 1.05, "risk": 0.8}
        if self.regime == "recession":
            return {"spend": 0.92, "risk": 1.35}
        return {"spend": 1.0, "risk": 1.0}

    # ---------- Two-stage decision logic ----------
    def _choose_discrete_action(
        self,
        *,
        is_cli_candidate: int,
        status: str,
        util: float,
        dpd12: int,
        pay_full: bool,
        overlimit_rate: float,
        hardship: int,
        jobless: int,
        macro_risk: float,
        external_score: float,
    ) -> int:
        score_n = clip01((external_score - 300.0) / 600.0)
        score_good = score_n
        score_bad = 1.0 - score_n

        if status != "current":
            risk = 0.0
            risk += 0.60 * (dpd12 / 12.0)
            risk += 0.25 * (1.0 if (hardship or jobless) else 0.0)
            risk += 0.20 * overlimit_rate
            risk *= macro_risk

            cld_prob = clip01(0.03 + 0.20 * risk + 0.08 * score_bad)
            if np.random.rand() < cld_prob:
                return ACTION_CLD
            return ACTION_HOLD

        good = 0.0
        good += 0.55 * (1.0 if pay_full else 0.0)
        good += 0.25 * clip01((0.85 - util) / 0.85)
        good += 0.20 * clip01(1.0 - dpd12 / 12.0)

        bad = 0.0
        bad += 0.55 * (dpd12 / 12.0)
        bad += 0.25 * overlimit_rate
        bad += 0.20 * (1.0 if (hardship or jobless) else 0.0)
        bad *= macro_risk

        if is_cli_candidate:
            cli_prob = (0.01 + 0.08 * good * (0.5 + 0.8 * score_good)) / max(1.0, macro_risk)
            if np.random.rand() < clip01(cli_prob):
                return ACTION_CLI

        cld_prob = 0.008 + 0.10 * bad * (0.6 + 0.9 * score_bad)
        if np.random.rand() < clip01(cld_prob):
            return ACTION_CLD

        return ACTION_HOLD

    def _compute_magnitude_pct(
        self,
        action_id: int,
        *,
        util: float,
        dpd12: int,
        pay_ratio: float,
        min_pay_ratio: float,
        external_score: float,
        overlimit_rate: float,
        hardship: int,
        jobless: int,
        macro_risk: float,
    ) -> float:
        if action_id == ACTION_HOLD:
            return 0.0

        dpd_n = dpd12 / 12.0
        score_n = clip01((external_score - 300.0) / 600.0)
        score_good = score_n
        score_bad = 1.0 - score_n

        pay_term = clip01(0.7 * pay_ratio + 0.3 * min_pay_ratio)

        if action_id == ACTION_CLI:
            util_term = 1.0 - abs(util - 0.60) / 0.60
            util_term = clip01(util_term)

            risk_term = clip01(1.0 - dpd_n) * clip01(1.0 - overlimit_rate)
            hardship_ok = 0.0 if (hardship or jobless) else 1.0

            base = (0.35 * util_term + 0.40 * pay_term + 0.25 * risk_term) * hardship_ok
            base = base / max(1.0, macro_risk)

            cibil_mult = 0.6 + 0.8 * score_good

            pct = self.oc.cli_pct_min + (base * cibil_mult) * (self.oc.cli_pct_max - self.oc.cli_pct_min)
            return float(np.clip(pct, self.oc.cli_pct_min, self.oc.cli_pct_max))

        if action_id == ACTION_CLD:
            bad_pay = clip01(1.0 - pay_ratio)
            bad_min = clip01(1.0 - min_pay_ratio)
            hardship_term = 1.0 if (hardship or jobless) else 0.0

            base = (
                0.45 * dpd_n
                + 0.20 * bad_pay
                + 0.15 * bad_min
                + 0.15 * clip01(util)
                + 0.05 * overlimit_rate
                + 0.10 * hardship_term
            )
            base = clip01(base) * macro_risk

            cibil_amp = 1.0 + 1.0 * score_bad

            pct = self.oc.cld_pct_min + (base * cibil_amp) * (self.oc.cld_pct_max - self.oc.cld_pct_min)
            return float(np.clip(pct, self.oc.cld_pct_min, self.oc.cld_pct_max))

        return 0.0

    # ---------- Bureau score drift ----------
    def _update_external_score_monthly(
        self, i: int, payment_ratio: float, min_pay_ratio: float, util: float
    ) -> Tuple[float, float]:
        score = float(self.customers.at[i, "external_score"])
        status = str(self.customers.at[i, "status"])
        dpd12 = int(self.dpd_count_12m[i])

        if status == "default":
            delta = -np.random.uniform(80, 130)
        elif status in ("dpd60", "dpd90"):
            delta = -np.random.uniform(25, 55)
        elif status == "dpd30":
            delta = -np.random.uniform(10, 25)
        else:
            good_pay = payment_ratio >= 0.95
            ok_pay = payment_ratio >= 0.75
            low_util = util <= 0.70
            high_util = util >= 0.90

            if dpd12 == 0 and good_pay and low_util:
                delta = np.random.uniform(2, 6)
            elif dpd12 == 0 and ok_pay:
                delta = np.random.uniform(0, 3)
            elif high_util:
                delta = -np.random.uniform(1, 5)
            elif dpd12 >= 2:
                delta = -np.random.uniform(3, 10)
            else:
                delta = np.random.uniform(-1, 1)

        new_score = float(np.clip(score + delta, 300.0, 900.0))
        self.customers.at[i, "external_score"] = new_score
        return new_score, float(delta)

    # ---------- Transactions ----------
    def _gen_transactions_for_month(self, ym: str, mstart: datetime, mend: datetime):
        month_idx = mstart.month - 1
        season_mult = self.cc.monthly_multipliers[month_idx]
        macro = self._macro_multipliers()

        lam = self.oc.avg_tx_per_cust_per_month * season_mult * macro["spend"]
        tx_counts = np.random.poisson(lam=lam, size=len(self.customers))

        def rand_ts():
            probs = np.array(
                [0.01, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.09,
                 0.09, 0.09, 0.09, 0.08, 0.06, 0.06, 0.06, 0.05, 0.04, 0.03, 0.02, 0.02]
            )
            probs /= probs.sum()
            hour = np.random.choice(np.arange(24), p=probs)
            minute = np.random.randint(0, 60)
            second = np.random.randint(0, 60)
            day = np.random.randint(mstart.day, mend.day + 1)
            return mstart.replace(day=day, hour=hour, minute=minute, second=second)

        mccs = [
            ("grocery", 1.0), ("fuel", 0.7), ("online_retail", 1.2), ("electronics", 2.8),
            ("restaurants", 1.1), ("travel", 3.5), ("utilities", 1.5), ("entertainment", 1.7),
        ]

        month_tx_sum = np.zeros(len(self.customers))
        month_tx_cnt = np.zeros(len(self.customers))
        month_overlimit_cnt = np.zeros(len(self.customers))

        rows = []
        for i, cust in self.customers.iterrows():
            if cust["status"] in ("default", "closed"):
                continue

            limit_local = float(cust["credit_limit"])
            bal_local = float(cust["balance"])

            # ROOT FIX B: create utilization spread by targeting utilization of limit
            util_target = np.clip(np.random.normal(loc=0.22, scale=0.12), 0.02, 0.85)
            desired_spend = util_target * limit_local * np.random.uniform(0.7, 1.2)

            available = max(0.0, limit_local - bal_local)
            monthly_budget = max(0.0, min(desired_spend, available * np.random.uniform(0.8, 1.10)))

            n_tx = tx_counts[i]
            if n_tx == 0 or monthly_budget <= 0:
                continue

            weights = np.random.dirichlet(alpha=np.ones(n_tx))

            for w in weights:
                mcc, scale = mccs[np.random.randint(len(mccs))]
                base = np.random.lognormal(mean=np.log(180 * scale), sigma=0.8)
                amount = min(monthly_budget * w * np.random.uniform(0.9, 1.3), base)
                if amount < 1.0:
                    continue
                ts = rand_ts()

                overlimit = 0
                if bal_local + amount > limit_local * 1.05 and np.random.rand() < 0.02:
                    overlimit = 1

                rows.append(
                    {
                        "cust_id": cust["cust_id"],
                        "event_time": ts.isoformat(sep=" "),
                        "amount": round(float(amount), 2),
                        "mcc": mcc,
                        "merchant_id": f"M{np.random.randint(1000, 9999)}",
                        "channel": np.random.choice(["POS", "ECOM", "ATM"], p=[0.62, 0.36, 0.02]),
                        "geo": np.random.choice(["IN-TS", "IN-TG", "IN-MH", "IN-KA", "IN-TN", "IN-DL"]),
                        "overlimit_attempt": overlimit,
                        "macro_regime": self.regime,
                        "year_month": ym,
                    }
                )

                bal_local += float(amount)
                month_tx_sum[i] += float(amount)
                month_tx_cnt[i] += 1.0
                month_overlimit_cnt[i] += float(overlimit)

            util_local = bal_local / max(1.0, limit_local)
            self.customers.at[i, "balance"] = bal_local
            self.customers.at[i, "utilization"] = util_local

        if rows:
            self.transactions.extend(rows)

        alpha_amt = 2 / (3 + 1)
        alpha_cnt = 2 / (1 + 1)
        alpha_ovl = 2 / (3 + 1)

        self.tx_amt_ema = (1 - alpha_amt) * self.tx_amt_ema + alpha_amt * month_tx_sum
        self.tx_cnt_ema = (1 - alpha_cnt) * self.tx_cnt_ema + alpha_cnt * month_tx_cnt

        rates = np.divide(month_overlimit_cnt, np.maximum(1, month_tx_cnt))
        self.overlimit_ema = (1 - alpha_ovl) * self.overlimit_ema + alpha_ovl * rates

        for _, cust in self.customers.iterrows():
            self.income_history.append(
                {
                    "cust_id": int(cust["cust_id"]),
                    "date": mstart.date().isoformat(),
                    "monthly_income": round(float(cust["monthly_income"]), 2),
                }
            )

    # ---------- Life events ----------
    def _apply_life_events(self, mstart: datetime):
        macro = self._macro_multipliers()
        for i, cust in self.customers.iterrows():
            if cust["status"] in ("default", "closed"):
                continue

            infl = self.mc.inflation_annual[self.regime]
            drift = np.random.normal(loc=(0.01 + infl) / 12.0, scale=0.02 / 12.0)
            new_income = max(1000.0, cust["monthly_income"] * (1.0 + drift))
            self.customers.at[i, "monthly_income"] = new_income

            job_hazard = 0.002 * macro["risk"]
            if not cust["jobless"] and np.random.rand() < job_hazard:
                self.customers.at[i, "jobless"] = 1
                self.customers.at[i, "monthly_income"] *= np.random.uniform(0.2, 0.6)

            if cust["jobless"] and np.random.rand() < 0.05:
                self.customers.at[i, "hardship"] = 1
                self.credit_events.append(
                    {
                        "cust_id": cust["cust_id"],
                        "event_date": mstart.date().isoformat(),
                        "event_type": "HARDSHIP_ON",
                        "action_id": None,
                        "magnitude_pct": None,
                        "pre_credit_limit": round(float(self.customers.at[i, "credit_limit"]), 2),
                        "delta": 0.0,
                        "new_credit_limit": round(float(self.customers.at[i, "credit_limit"]), 2),
                        "reason": "Job loss",
                    }
                )
            elif cust["hardship"] and np.random.rand() < 0.02:
                self.customers.at[i, "hardship"] = 0
                self.credit_events.append(
                    {
                        "cust_id": cust["cust_id"],
                        "event_date": mstart.date().isoformat(),
                        "event_type": "HARDSHIP_OFF",
                        "action_id": None,
                        "magnitude_pct": None,
                        "pre_credit_limit": round(float(self.customers.at[i, "credit_limit"]), 2),
                        "delta": 0.0,
                        "new_credit_limit": round(float(self.customers.at[i, "credit_limit"]), 2),
                        "reason": "Recovery",
                    }
                )

    # ---------- Statements & collections ----------
    def _process_statements(self, mstart: datetime, mend: datetime):
        stmt_date = mstart.replace(day=min(self.oc.statement_day, mend.day))
        macro = self._macro_multipliers()

        for i, cust in self.customers.iterrows():
            if cust["status"] in ("default", "closed"):
                continue

            bal = float(cust["balance"])
            pre_limit = float(cust["credit_limit"])

            # ROOT FIX C: pre-payment utilization is the demand signal
            util_pre = bal / max(1.0, pre_limit)
            util_pre = float(np.clip(util_pre, 0.0, 5.0))

            interest = bal * self.oc.apr_annual / 12.0 if bal > 1.0 else 0.0
            min_due = max(self.oc.min_pay_amount_floor, bal * self.oc.min_pay_rate) if bal > 0 else 0.0

            overlimit_fee = self.oc.overlimit_fee if bal > pre_limit else 0.0
            late_fee = 0.0

            pay_full_prob = clip01(cust["pay_full_prob"] / (macro["risk"] ** 0.25))
            pay_full = np.random.rand() < pay_full_prob

            # -------------------------------
            # ROOT FIX A: revolve payments must NOT be overwritten
            # -------------------------------
            payment = 0.0
            if bal > 0:
                if pay_full:
                    payment = bal + interest + overlimit_fee
                else:
                    revolve = float(cust.get("revolve_prop", 0.3))

                    base_pay = min_due * np.random.uniform(0.8, 1.1)
                    extra_pay = revolve * max(0.0, (bal - min_due)) * np.random.uniform(0.10, 0.40)

                    payment = base_pay + extra_pay

                    if np.random.rand() < (self.cc.roll_rate_base * macro["risk"]):
                        new_status = {
                            "current": "dpd30",
                            "dpd30": "dpd60",
                            "dpd60": "dpd90",
                            "dpd90": "default",
                        }.get(cust["status"], "default")
                        self.customers.at[i, "status"] = new_status
                        late_fee = self.oc.late_fee

            total_due = bal + interest + overlimit_fee + late_fee
            payment = min(payment, total_due)
            new_balance = max(0.0, total_due - payment)

            # Cure chance
            if self.customers.at[i, "status"] in ("dpd30", "dpd60") and np.random.rand() < (
                self.cc.cure_rate_base / macro["risk"]
            ):
                self.customers.at[i, "status"] = "current"

            # Default hazard uses post-payment utilization
            util_post = new_balance / max(1.0, pre_limit)
            hazard = self.cc.default_hazard_base * macro["risk"] * (1.0 + 1.8 * min(1.5, util_post))
            if np.random.rand() < hazard:
                self.customers.at[i, "status"] = "default"

            # Churn hazard
            churn_adj = self.cc.churn_hazard_base * (0.5 + 0.5 * (1.0 - util_post)) * (1.2 if pay_full else 0.8)
            closed_now = False
            if np.random.rand() < churn_adj and self.customers.at[i, "status"] == "current":
                self.customers.at[i, "status"] = "closed"
                closed_now = True
                new_balance = 0.0
                util_post = 0.0

            # Update state
            self.customers.at[i, "balance"] = new_balance
            self.customers.at[i, "utilization"] = util_post
            self.customers.at[i, "months_on_book"] += 1
            self.customers.at[i, "last_stmt_date"] = stmt_date

            # dpd counter
            if self.customers.at[i, "status"] in ("dpd30", "dpd60", "dpd90"):
                self.dpd_count_12m[i] = min(12, self.dpd_count_12m[i] + 1)
            else:
                self.dpd_count_12m[i] = max(0, self.dpd_count_12m[i] - 1)

            # ROOT FIX D: rolling window tracks util_pre (demand), not util_post
            self.util_windows[int(cust["cust_id"])].append(util_pre)
            max_util_6m = max(self.util_windows[int(cust["cust_id"])]) if self.util_windows[int(cust["cust_id"])] else 0.0

            # Candidate flags using util_pre
            is_cli_candidate = int(
                (self.customers.at[i, "status"] == "current")
                and (self.customers.at[i, "months_on_book"] >= 6)
                and (
                    (util_pre >= 0.10 and util_pre <= 0.90)
                    or (max_util_6m >= 0.15)
                    or (self.tx_cnt_ema[i] >= 8)
                )
            )
            is_cld_candidate = int(self.customers.at[i, "status"] in ("current", "dpd30", "dpd60", "dpd90"))

            # Ratios
            payment_ratio = (payment / total_due) if total_due > 0 else 0.0
            min_pay_ratio = (payment / max(1e-6, min_due)) if min_due > 0 else 0.0

            # Monthly bureau drift uses util_post (risk signal)
            new_score, score_delta = self._update_external_score_monthly(i, payment_ratio, min_pay_ratio, util_post)

            # Decision uses util_pre (demand)
            action_id = self._choose_discrete_action(
                is_cli_candidate=is_cli_candidate,
                status=str(self.customers.at[i, "status"]),
                util=float(util_pre),
                dpd12=int(self.dpd_count_12m[i]),
                pay_full=bool(pay_full),
                overlimit_rate=float(self.overlimit_ema[i]),
                hardship=int(self.customers.at[i, "hardship"]),
                jobless=int(self.customers.at[i, "jobless"]),
                macro_risk=float(macro["risk"]),
                external_score=float(new_score),
            )

            mag_pct = self._compute_magnitude_pct(
                action_id,
                util=float(util_pre),
                dpd12=int(self.dpd_count_12m[i]),
                pay_ratio=float(payment_ratio),
                min_pay_ratio=float(min_pay_ratio),
                external_score=float(new_score),
                overlimit_rate=float(self.overlimit_ema[i]),
                hardship=int(self.customers.at[i, "hardship"]),
                jobless=int(self.customers.at[i, "jobless"]),
                macro_risk=float(macro["risk"]),
            )

            event_type = "HOLD"
            delta_amt = 0.0

            if action_id == ACTION_CLI and is_cli_candidate:
                event_type = "CLI"
                delta_amt = pre_limit * mag_pct
                post_limit = min(self.oc.credit_limit_cap, pre_limit + delta_amt)
                self.customers.at[i, "credit_limit"] = post_limit
                self.last_cli_limit[i] = post_limit

            elif action_id == ACTION_CLD and is_cld_candidate:
                event_type = "CLD"
                delta_amt = -pre_limit * mag_pct
                post_limit = max(self.oc.credit_limit_floor, pre_limit + delta_amt)
                self.customers.at[i, "credit_limit"] = post_limit

            else:
                action_id = ACTION_HOLD
                mag_pct = 0.0
                delta_amt = 0.0
                post_limit = pre_limit

            if event_type != "HOLD" or self.oc.log_hold_events:
                self.credit_events.append(
                    {
                        "cust_id": cust["cust_id"],
                        "event_date": stmt_date.date().isoformat(),
                        "event_type": event_type,
                        "action_id": int(action_id),
                        "magnitude_pct": round(float(mag_pct), 6),
                        "pre_credit_limit": round(float(pre_limit), 2),
                        "delta": round(float(delta_amt), 2),
                        "new_credit_limit": round(float(post_limit), 2),
                        "reason": "Two-stage: action then computed magnitude",
                    }
                )

            revenue = interest + late_fee + overlimit_fee
            risk_penalty = 0.0
            if self.customers.at[i, "status"] == "default":
                risk_penalty = 0.8 * new_balance
            churn_penalty = 50.0 if closed_now else 0.0

            self.statements.append(
                {
                    "cust_id": cust["cust_id"],
                    "statement_date": stmt_date.date().isoformat(),
                    "prev_balance": round(float(bal), 2),
                    "interest": round(float(interest), 2),
                    "late_fee": round(float(late_fee), 2),
                    "overlimit_fee": round(float(overlimit_fee), 2),
                    "util_pre": round(float(util_pre), 4),
                    "util_post": round(float(util_post), 4),
                    "payment": round(float(payment), 2),
                    "min_due": round(float(min_due), 2),
                    "payment_ratio": round(float(payment_ratio), 4),
                    "min_pay_ratio": round(float(min_pay_ratio), 4),
                    "new_balance": round(float(new_balance), 2),
                    "status": self.customers.at[i, "status"],
                    "credit_limit": round(float(self.customers.at[i, "credit_limit"]), 2),
                    "is_cli_candidate": int(is_cli_candidate),
                    "macro_regime": self.regime,
                    "revenue": round(float(revenue), 2),
                    "risk_penalty": round(float(risk_penalty), 2),
                    "churn_penalty": round(float(churn_penalty), 2),
                    "action_id": int(action_id),
                    "action_name": ACTION_NAME[int(action_id)],
                    "magnitude_pct": round(float(mag_pct), 6),
                }
            )

            recent_cli_effectiveness = float(self.tx_amt_ema[i] / max(1.0, self.last_cli_limit[i]))
            self.derived_monthly.append(
                {
                    "cust_id": cust["cust_id"],
                    "date": stmt_date.date().isoformat(),
                    "avg_tx_amt_90d": round(float(self.tx_amt_ema[i]), 2),
                    "tx_count_30d": round(float(self.tx_cnt_ema[i]), 2),
                    "overlimit_rate_90d": round(float(self.overlimit_ema[i]), 4),
                    "max_utilization_6m": round(float(max_util_6m), 4),
                    "dpd_count_12m": int(self.dpd_count_12m[i]),
                    "recent_cli_effectiveness": round(recent_cli_effectiveness, 6),
                    "external_score": round(float(new_score), 2),
                    "external_score_delta": round(float(score_delta), 2),
                }
            )

    # ---------- Run loop ----------
    def run(self):
        os.makedirs(self.oc.out_dir, exist_ok=True)
        dt = month_start(self.start_date)
        while dt < self.end_date:
            ym = year_month_key(dt)

            self._advance_macro()
            self._gen_transactions_for_month(ym, dt, month_end(dt))
            self._apply_life_events(dt)
            self._process_statements(dt, month_end(dt))

            if self.oc.partition_by_year and dt.month == 12:
                self._flush_year(dt.year)
                self.transactions.clear()
                self.statements.clear()
                self.credit_events.clear()
                self.income_history.clear()
                self.derived_monthly.clear()

            dt = month_start(dt + timedelta(days=32))

        self._flush_year(self.end_date.year)
        self.customers.to_csv(os.path.join(self.oc.out_dir, "customers.csv"), index=False)
        self._write_action_map()

    def _write_action_map(self):
        path = os.path.join(self.oc.out_dir, "action_map.csv")
        df = pd.DataFrame(
            [
                {"action_id": ACTION_HOLD, "action_name": "HOLD"},
                {"action_id": ACTION_CLI, "action_name": "CLI"},
                {"action_id": ACTION_CLD, "action_name": "CLD"},
            ]
        )
        df.to_csv(path, index=False)

    def _flush_year(self, year: int):
        def write(df: pd.DataFrame, name: str):
            if df.empty:
                return
            df.to_csv(os.path.join(self.oc.out_dir, f"{name}_{year}.csv"), index=False)

        if self.transactions:
            df_tx = pd.DataFrame(self.transactions)
            write(df_tx[df_tx["event_time"].str[:4] == str(year)], "transactions")

        if self.statements:
            df_stmt = pd.DataFrame(self.statements)
            write(df_stmt[df_stmt["statement_date"].str[:4] == str(year)], "statements")

        if self.credit_events:
            df_ce = pd.DataFrame(self.credit_events)
            write(df_ce[df_ce["event_date"].str[:4] == str(year)], "credit_events")

        if self.income_history:
            df_inc = pd.DataFrame(self.income_history)
            write(df_inc[df_inc["date"].str[:4] == str(year)], "income_history")

        if self.derived_monthly:
            df_drv = pd.DataFrame(self.derived_monthly)
            write(df_drv[df_drv["date"].str[:4] == str(year)], "derived_features")

    # ---------- Realtime (optional) ----------
    def realtime_events(self) -> Iterator[Dict]:
        dt = month_start(self.start_date)
        while dt < self.end_date:
            ym = year_month_key(dt)
            self._advance_macro()
            self._gen_transactions_for_month(ym, dt, month_end(dt))
            for row in sorted(self.transactions, key=lambda r: r["event_time"]):
                yield {"topic": "transactions", "payload": row}
                if self.oc.realtime_sleep > 0:
                    time.sleep(self.oc.realtime_sleep)
            self._apply_life_events(dt)
            self._process_statements(dt, month_end(dt))
            for row in self.statements:
                yield {"topic": "statements", "payload": row}
            for row in self.credit_events:
                yield {"topic": "credit_events", "payload": row}
            self.transactions.clear()
            self.statements.clear()
            self.credit_events.clear()
            self.income_history.clear()
            self.derived_monthly.clear()
            dt = month_start(dt + timedelta(days=32))


# ---------------------------------
# CLI
# ---------------------------------

def main():
    import argparse

    p = argparse.ArgumentParser(description="Advanced finance simulator (v2) with 3-action RL + computed magnitude + monthly bureau drift")
    p.add_argument("--customers", type=int, default=20000)
    p.add_argument("--years", type=int, default=20)
    p.add_argument("--start", type=str, default="2006-01-01")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="sim_output_v2")
    p.add_argument("--no-partition", action="store_true")
    p.add_argument("--realtime", action="store_true")
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--avg-tx-per-month", type=float, default=18.0)
    p.add_argument("--log-hold-events", action="store_true", help="Also write HOLD rows into credit_events.")

    args = p.parse_args()

    cconf = CustomerConfig(n_customers=args.customers, years=args.years, start_date=args.start, seed=args.seed)
    mconf = MacroConfig()
    oconf = OutputConfig(
        out_dir=args.out,
        partition_by_year=not args.no_partition,
        realtime_sleep=args.sleep,
        avg_tx_per_cust_per_month=args.avg_tx_per_month,
        log_hold_events=args.log_hold_events,
    )

    sim = FinanceSimulatorV2(cconf, mconf, oconf)

    if args.realtime:
        for event in sim.realtime_events():
            print(json.dumps(event, ensure_ascii=False))
    else:
        sim.run()
        print(f"[v2-rl] Simulation complete. Files written to: {args.out}")

if __name__ == "__main__":
    main()
