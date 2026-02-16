import os
import json
import time
import json
from pathlib import Path
from datetime import datetime, timezone
import logging
import sys
import re
from statistics import median

import requests

"""Agentic Drift Supervisor (multi-action, deterministic)

This script implements a *supervisory agent* loop for your dissertation demo:

Perceive  -> reads latest eval artifacts (metrics.json + drift.json)
Reason    -> computes severity + compares against rolling baseline
Plan      -> selects best remediation: WAIT | RECALIBRATE | RETRAIN_GATE | RETRAIN_DIR | RETRAIN_MAG | RETRAIN_FULL
Act       -> triggers Airflow DAG run with an action/scope payload
Reflect   -> stores evidence history + baseline + last decisions to avoid flapping

LLM is intentionally NOT used for decision-making.
"""


# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("drift-agent")


# ---------------------------
# Config via env vars
# ---------------------------
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "/shared/reports"))
EVAL_DIR = Path(os.getenv("EVAL_DIR", str(REPORTS_DIR / "eval")))
STATE_PATH = Path(os.getenv("STATE_PATH", str(REPORTS_DIR / "agent_state.json")))

AIRFLOW_BASE_URL = os.getenv("AIRFLOW_BASE_URL", "http://airflow:8080")
AIRFLOW_DAG_ID = os.getenv("AIRFLOW_DAG_ID", "mlflow_train_taskflow_optionA")
AIRFLOW_USERNAME = os.getenv("AIRFLOW_USERNAME", "airflow")
AIRFLOW_PASSWORD = os.getenv("AIRFLOW_PASSWORD", "airflow")
AIRFLOW_AUTH_MODE = os.getenv("AIRFLOW_AUTH_MODE", "session").lower()  # session | basic

POLL_SECONDS = int(os.getenv("POLL_SECONDS", "30"))

# Memory / anti-flap
EVIDENCE_WINDOW = int(os.getenv("EVIDENCE_WINDOW", "6"))
MIN_CONSECUTIVE = int(os.getenv("MIN_CONSECUTIVE", "2"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "900"))  # 15 min

# Drift thresholds
THRESH_ACTION_JS = float(os.getenv("THRESH_ACTION_JS", "0.10"))
THRESH_ACTION_JS_CRIT = float(os.getenv("THRESH_ACTION_JS_CRIT", "0.20"))

THRESH_MAX_FEATURE_PSI = float(os.getenv("THRESH_MAX_FEATURE_PSI", "0.25"))
THRESH_NUM_FEATURES_PSI = int(os.getenv("THRESH_NUM_FEATURES_PSI", "5"))
THRESH_NUM_FEATURES_PSI_CRIT = int(os.getenv("THRESH_NUM_FEATURES_PSI_CRIT", "10"))

# Performance thresholds (relative drop vs baseline)
THRESH_GATE_F1_DROP = float(os.getenv("THRESH_GATE_F1_DROP", "0.05"))
THRESH_DIR_F1_DROP = float(os.getenv("THRESH_DIR_F1_DROP", "0.05"))

# Calibration thresholds
THRESH_GATE_BRIER = float(os.getenv("THRESH_GATE_BRIER", "0.12"))
THRESH_GATE_ECE = float(os.getenv("THRESH_GATE_ECE", "0.20"))

# Magnitude thresholds (relative rise vs baseline)
THRESH_MAG_MAE_RISE = float(os.getenv("THRESH_MAG_MAE_RISE", "0.15"))
THRESH_MAG_P95_RISE = float(os.getenv("THRESH_MAG_P95_RISE", "0.25"))


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def load_state() -> dict:
    if STATE_PATH.exists():
        return load_json(STATE_PATH)
    return {
        "last_seen_eval_dir": None,
        "last_triggered_at": None,
        "last_action": None,
        "last_reasons": None,
        "history": [],  # list[EvidenceSnapshot dict]
        "consecutive": {},  # action -> int
        "baseline": {},
    }


def find_latest_eval_run_dir(eval_root: Path) -> Path | None:
    """Supports both:
    - reports/eval/metrics.json
    - reports/eval/<run_id>/metrics.json
    """
    if (eval_root / "metrics.json").exists() and (eval_root / "drift.json").exists():
        return eval_root

    subdirs = [p for p in eval_root.glob("*") if p.is_dir()]
    if not subdirs:
        return None

    subdirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    candidate = subdirs[0]
    if (candidate / "metrics.json").exists() and (candidate / "drift.json").exists():
        return candidate
    return None


def _safe_float(x):
    try:
        if x is None:
            return None
        x = float(x)
        if x != x:
            return None
        return x
    except Exception:
        return None


def _get(d: dict, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def llm_audit_note(plan: dict, snap: dict) -> dict | None:
    """Returns validated JSON audit note or None (caller uses deterministic fallback)."""
    if os.getenv("LLM_ENABLED", "false").lower() != "true":
        return None

    url = os.getenv("LLM_URL", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()
    if not url or not model:
        return None

    timeout = int(float(os.getenv("LLM_TIMEOUT_SECONDS", "20")))
    max_chars = int(float(os.getenv("LLM_MAX_CHARS", "1200")))

    # Only give the LLM what it needs (no free-form logs!)
    evidence = {
        "chosen_action": plan.get("action"),
        "scope": plan.get("scope"),
        "severity": plan.get("severity"),
        "reasons": plan.get("reasons", []),
        "snapshot": {
            "action_js": snap.get("action_js"),
            "n_feat_psi_hi": snap.get("n_feat_psi_hi"),
            "top_psi": snap.get("top_psi", [])[:5],
            "gate_f1": snap.get("gate_f1"),
            "gate_pr_auc": snap.get("gate_pr_auc"),
            "gate_brier": snap.get("gate_brier"),
            "gate_ece": snap.get("gate_ece"),
            "dir_f1": snap.get("dir_f1"),
            "cli_mae": snap.get("cli_mae"),
            "cli_p95": snap.get("cli_p95"),
            "cld_mae": snap.get("cld_mae"),
            "cld_p95": snap.get("cld_p95"),
            "hold_rate": snap.get("hold_rate"),
            "entropy": snap.get("entropy"),
        }
    }

    prompt = f"""
You are an audit assistant for an ML retraining supervisor.
You MUST NOT change the decision. Only explain it.

Return ONLY valid JSON with keys:
- "summary" (<= 80 words)
- "evidence_used" (array of short strings)
- "risk_note" (<= 40 words)
- "next_steps" (array of 2-4 short strings)

Decision (DO NOT CHANGE):
{json.dumps(evidence, ensure_ascii=False)}
""".strip()

    payload = {"model": model, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    txt = r.json().get("response", "")

    # Hard cut to avoid runaway
    txt = txt[:max_chars]

    # Extract JSON object safely
    m = re.search(r"\{.*\}", txt, flags=re.S)
    if not m:
        return None

    try:
        out = json.loads(m.group(0))
    except Exception:
        return None

    # Minimal schema validation
    required = ["summary", "evidence_used", "risk_note", "next_steps"]
    if not all(k in out for k in required):
        return None
    if not isinstance(out["evidence_used"], list) or not isinstance(out["next_steps"], list):
        return None

    return out



def build_snapshot(metrics: dict, drift: dict, source_eval_dir: str) -> dict:
    """Flatten only what we need for decisions + baseline."""

    # Drift
    action_js = _safe_float(drift.get("action_js_divergence"))
    feat_drift = drift.get("feature_drift", {}) or {}
    psis = []
    for feat, info in feat_drift.items():
        psi = _safe_float((info or {}).get("psi"))
        if psi is not None:
            psis.append((feat, psi))
    psis.sort(key=lambda x: x[1], reverse=True)
    high = [(f, v) for (f, v) in psis if v > THRESH_MAX_FEATURE_PSI]

    # Gate
    gate_f1 = _safe_float(_get(metrics, "gate", "f1"))
    gate_pr_auc = _safe_float(_get(metrics, "gate", "pr_auc"))
    gate_brier = _safe_float(_get(metrics, "gate", "brier"))
    gate_ece = _safe_float(_get(metrics, "gate", "ece_10bins"))

    # Dir
    dir_f1 = _safe_float(_get(metrics, "dir", "f1"))

    # Magnitude
    cli_mae = _safe_float(_get(metrics, "magnitude", "CLI", "mae_pp"))
    cli_p95 = _safe_float(_get(metrics, "magnitude", "CLI", "p95"))
    cld_mae = _safe_float(_get(metrics, "magnitude", "CLD", "mae_pp"))
    cld_p95 = _safe_float(_get(metrics, "magnitude", "CLD", "p95"))

    # Policy behavior
    hold_rate = _safe_float(_get(metrics, "action_behavior", "pred_action_dist", "HOLD"))
    cli_rate = _safe_float(_get(metrics, "action_behavior", "pred_action_dist", "CLI"))
    cld_rate = _safe_float(_get(metrics, "action_behavior", "pred_action_dist", "CLD"))
    entropy = _safe_float(_get(metrics, "action_behavior", "pred_action_entropy"))

    return {
        "ts": now_utc_iso(),
        "source_eval_dir": source_eval_dir,
        "action_js": action_js,
        "n_feat_psi_hi": len(high),
        "top_psi": psis[:10],
        "gate_f1": gate_f1,
        "gate_pr_auc": gate_pr_auc,
        "gate_brier": gate_brier,
        "gate_ece": gate_ece,
        "dir_f1": dir_f1,
        "cli_mae": cli_mae,
        "cli_p95": cli_p95,
        "cld_mae": cld_mae,
        "cld_p95": cld_p95,
        "hold_rate": hold_rate,
        "cli_rate": cli_rate,
        "cld_rate": cld_rate,
        "entropy": entropy,
    }


def _median(values):
    vals = [v for v in values if isinstance(v, (int, float)) and v == v]
    if not vals:
        return None
    return float(median(vals))


def compute_baseline(history: list[dict]) -> dict:
    """Rolling baseline = median over last EVIDENCE_WINDOW snapshots."""
    recent = history[-EVIDENCE_WINDOW:] if len(history) > EVIDENCE_WINDOW else history
    keys = [
        "action_js",
        "n_feat_psi_hi",
        "gate_f1",
        "gate_pr_auc",
        "gate_brier",
        "gate_ece",
        "dir_f1",
        "cli_mae",
        "cli_p95",
        "cld_mae",
        "cld_p95",
        "hold_rate",
        "entropy",
    ]
    base = {}
    for k in keys:
        base[k] = _median([s.get(k) for s in recent])
    return base


def rel_drop(cur: float | None, base: float | None) -> float | None:
    if cur is None or base is None or base == 0:
        return None
    return max(0.0, (base - cur) / abs(base))


def rel_rise(cur: float | None, base: float | None) -> float | None:
    if cur is None or base is None or base == 0:
        return None
    return max(0.0, (cur - base) / abs(base))


def plan_action(snapshot: dict, baseline: dict) -> tuple[str, str, list[str], dict]:
    """Returns (action, scope, reasons, debug).

    Actions:
      WAIT | RECALIBRATE | RETRAIN_GATE | RETRAIN_DIR | RETRAIN_MAG | RETRAIN_FULL
    """

    reasons: list[str] = []

    # Drift signals
    action_js = snapshot.get("action_js")
    npsi = snapshot.get("n_feat_psi_hi")
    drift_crit = (
        (action_js is not None and action_js >= THRESH_ACTION_JS_CRIT)
        or (isinstance(npsi, int) and npsi >= THRESH_NUM_FEATURES_PSI_CRIT)
    )
    drift_hi = (
        (action_js is not None and action_js >= THRESH_ACTION_JS)
        or (isinstance(npsi, int) and npsi >= THRESH_NUM_FEATURES_PSI)
    )

    # Performance deltas vs baseline
    gate_f1_drop = rel_drop(snapshot.get("gate_f1"), baseline.get("gate_f1"))
    dir_f1_drop = rel_drop(snapshot.get("dir_f1"), baseline.get("dir_f1"))

    # Magnitude deltas vs baseline
    cli_mae_rise = rel_rise(snapshot.get("cli_mae"), baseline.get("cli_mae"))
    cld_mae_rise = rel_rise(snapshot.get("cld_mae"), baseline.get("cld_mae"))
    cli_p95_rise = rel_rise(snapshot.get("cli_p95"), baseline.get("cli_p95"))
    cld_p95_rise = rel_rise(snapshot.get("cld_p95"), baseline.get("cld_p95"))

    # Calibration
    gate_brier = snapshot.get("gate_brier")
    gate_ece = snapshot.get("gate_ece")
    cal_bad = (
        (gate_brier is not None and gate_brier > THRESH_GATE_BRIER)
        or (gate_ece is not None and gate_ece > THRESH_GATE_ECE)
    )

    # Flags for planning
    gate_bad = (gate_f1_drop is not None and gate_f1_drop >= THRESH_GATE_F1_DROP)
    dir_bad = (dir_f1_drop is not None and dir_f1_drop >= THRESH_DIR_F1_DROP)
    mag_bad = (
        (cli_mae_rise is not None and cli_mae_rise >= THRESH_MAG_MAE_RISE)
        or (cld_mae_rise is not None and cld_mae_rise >= THRESH_MAG_MAE_RISE)
        or (cli_p95_rise is not None and cli_p95_rise >= THRESH_MAG_P95_RISE)
        or (cld_p95_rise is not None and cld_p95_rise >= THRESH_MAG_P95_RISE)
    )

    # Build reasons
    if drift_hi:
        reasons.append(
            f"drift_high(action_js={action_js}, n_feat_psi_hi={npsi}, top_psi={snapshot.get('top_psi', [])[:3]})"
        )
    if gate_bad:
        reasons.append(f"gate_f1_drop={gate_f1_drop:.3f} >= {THRESH_GATE_F1_DROP}")
    if dir_bad:
        reasons.append(f"dir_f1_drop={dir_f1_drop:.3f} >= {THRESH_DIR_F1_DROP}")
    if mag_bad:
        reasons.append(
            f"magnitude_shift(cli_mae_rise={cli_mae_rise}, cld_mae_rise={cld_mae_rise}, cli_p95_rise={cli_p95_rise}, cld_p95_rise={cld_p95_rise})"
        )
    if cal_bad and not gate_bad:
        reasons.append(f"calibration_shift(gate_brier={gate_brier}, gate_ece={gate_ece})")

    debug = {
        "baseline": baseline,
        "snapshot": snapshot,
        "deltas": {
            "gate_f1_drop": gate_f1_drop,
            "dir_f1_drop": dir_f1_drop,
            "cli_mae_rise": cli_mae_rise,
            "cld_mae_rise": cld_mae_rise,
            "cli_p95_rise": cli_p95_rise,
            "cld_p95_rise": cld_p95_rise,
        },
    }

    # Planning policy (priority-ordered, deterministic)
    # 1) Critical drift + any performance degradation -> full retrain
    if drift_crit and (gate_bad or dir_bad or mag_bad):
        return "RETRAIN_FULL", "full", reasons or ["critical_drift_plus_perf"], debug

    # 2) Critical drift alone -> full retrain (data distribution changed)
    if drift_crit:
        return "RETRAIN_FULL", "full", reasons or ["critical_distribution_shift"], debug

    # 3) Gate degradation -> retrain gate
    if gate_bad:
        return "RETRAIN_GATE", "gate_only", reasons or ["gate_degradation"], debug

    # 4) Direction degradation -> retrain direction
    if dir_bad:
        return "RETRAIN_DIR", "dir_only", reasons or ["dir_degradation"], debug

    # 5) Magnitude tail/MAE shift -> retrain magnitude
    if mag_bad:
        return "RETRAIN_MAG", "mag_only", reasons or ["magnitude_shift"], debug

    # 6) Calibration only -> recalibrate (fast remediation)
    if cal_bad:
        return "RECALIBRATE", "calibration", reasons or ["calibration_shift"], debug

    # 7) High (but not critical) drift -> conservative full retrain
    if drift_hi:
        return "RETRAIN_FULL", "full", reasons or ["distribution_shift"], debug

    return "WAIT", "none", reasons or ["stable"], debug


# ---------------------------
# Airflow actuation (Basic Auth OR UI Session Auth with CSRF)
# ---------------------------
_AIRFLOW_SESSION = None  # cached requests.Session (cookies in-memory)
_AIRFLOW_CSRF = None  # last csrf token (best-effort)


def _airflow_login_session() -> requests.Session:
    """UI-style login flow (CSRF-protected)."""
    global _AIRFLOW_CSRF
    base = AIRFLOW_BASE_URL.rstrip("/")
    s = requests.Session()

    r1 = s.get(f"{base}/login/", timeout=20)
    r1.raise_for_status()
    m = re.search(r'name="csrf_token"\s+type="hidden"\s+value="([^"]+)"', r1.text)
    if not m:
        raise RuntimeError("Could not extract csrf_token from Airflow /login/ page")
    csrf = m.group(1)
    _AIRFLOW_CSRF = csrf

    payload = {"username": AIRFLOW_USERNAME, "password": AIRFLOW_PASSWORD, "csrf_token": csrf}
    headers = {"Referer": f"{base}/login/", "Content-Type": "application/x-www-form-urlencoded"}
    r2 = s.post(f"{base}/login/", data=payload, headers=headers, allow_redirects=False, timeout=20)
    if r2.status_code not in (302, 200):
        raise RuntimeError(f"Airflow login failed: {r2.status_code} {r2.text[:200]}")

    # Verify session can access protected API
    r3 = s.get(f"{base}/api/v1/dags?limit=1", timeout=20)
    if r3.status_code != 200:
        raise RuntimeError(f"Airflow session auth check failed: {r3.status_code} {r3.text[:200]}")

    return s


def _trigger_airflow_basic(payload: dict) -> dict:
    url = f"{AIRFLOW_BASE_URL.rstrip('/')}/api/v1/dags/{AIRFLOW_DAG_ID}/dagRuns"
    r = requests.post(url, json=payload, auth=(AIRFLOW_USERNAME, AIRFLOW_PASSWORD), timeout=20)
    if r.status_code >= 300:
        raise RuntimeError(f"Airflow trigger failed (basic): {r.status_code} {r.text}")
    return r.json()


def deterministic_audit_note(plan: dict, snap: dict) -> dict:
    return {
        "summary": f"Action {plan.get('action')} selected based on drift/performance signals.",
        "evidence_used": plan.get("reasons", []),
        "risk_note": "Deterministic policy applied for auditability.",
        "next_steps": [
            "Monitor next cycle",
            "Validate retraining metrics post-deployment",
        ],
    }

def _trigger_airflow_session(payload: dict) -> dict:
    global _AIRFLOW_SESSION
    url = f"{AIRFLOW_BASE_URL.rstrip('/')}/api/v1/dags/{AIRFLOW_DAG_ID}/dagRuns"

    if _AIRFLOW_SESSION is None:
        _AIRFLOW_SESSION = _airflow_login_session()

    headers = {"Content-Type": "application/json"}
    if _AIRFLOW_CSRF:
        headers["X-CSRFToken"] = _AIRFLOW_CSRF
        headers["X-CSRF-Token"] = _AIRFLOW_CSRF

    r = _AIRFLOW_SESSION.post(url, json=payload, headers=headers, timeout=30)
    if r.status_code == 401:
        _AIRFLOW_SESSION = _airflow_login_session()
        if _AIRFLOW_CSRF:
            headers["X-CSRFToken"] = _AIRFLOW_CSRF
            headers["X-CSRF-Token"] = _AIRFLOW_CSRF
        r = _AIRFLOW_SESSION.post(url, json=payload, headers=headers, timeout=30)

    if r.status_code >= 300:
        raise RuntimeError(f"Airflow trigger failed (session): {r.status_code} {r.text}")
    return r.json()


def trigger_airflow_dag(conf: dict) -> dict:
    """Triggers a DAG run via Airflow REST API.

    Uses AIRFLOW_AUTH_MODE:
      - session (default): UI session cookie w/ CSRF login (no cookie.txt)
      - basic: HTTP basic auth
    """
    payload = {
        "dag_run_id": f"agentic__{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}",
        "conf": conf,
    }
    mode = (AIRFLOW_AUTH_MODE or "session").strip().lower()
    if mode == "basic":
        return _trigger_airflow_basic(payload)
    return _trigger_airflow_session(payload)


def in_cooldown(state: dict) -> bool:
    last = state.get("last_triggered_at")
    if not last:
        return False
    try:
        t = datetime.fromisoformat(last.replace("Z", "+00:00"))
    except Exception:
        return False
    delta = (datetime.now(timezone.utc) - t).total_seconds()
    return delta < COOLDOWN_SECONDS


def _inc_consecutive(state: dict, action: str) -> int:
    cons = state.get("consecutive", {}) or {}
    cons[action] = int(cons.get(action, 0)) + 1
    # reset other actions (prevents flapping between different actions)
    for k in list(cons.keys()):
        if k != action:
            cons[k] = 0
    state["consecutive"] = cons
    return cons[action]


def main():
    log.info("Agentic Drift Supervisor started")
    log.info(f"[agent] watching EVAL_DIR={EVAL_DIR}")
    state = load_state()

    while True:
        try:
            latest_dir = find_latest_eval_run_dir(EVAL_DIR)
            if latest_dir is None:
                time.sleep(POLL_SECONDS)
                continue

            latest_dir_str = str(latest_dir.resolve())
            if state.get("last_seen_eval_dir") == latest_dir_str:
                time.sleep(POLL_SECONDS)
                continue

            metrics_path = latest_dir / "metrics.json"
            drift_path = latest_dir / "drift.json"
            metrics = load_json(metrics_path)
            drift = load_json(drift_path)

            snap = build_snapshot(metrics, drift, latest_dir_str)

            # Reflect: update history
            hist = state.get("history", []) or []
            hist.append(snap)
            if len(hist) > max(50, EVIDENCE_WINDOW * 10):
                hist = hist[-max(50, EVIDENCE_WINDOW * 10) :]
            state["history"] = hist
            baseline = compute_baseline(hist)
            state["baseline"] = baseline

            action, scope, reasons, debug = plan_action(snap, baseline)
            state["last_seen_eval_dir"] = latest_dir_str

            log.info(f"[agent] plan={action} scope={scope} reasons={reasons}")

            # WAIT: no actuation
            if action == "WAIT":
                state["last_action"] = action
                state["last_reasons"] = reasons
                save_state(state)
                time.sleep(POLL_SECONDS)
                continue

            # Anti-flap: require MIN_CONSECUTIVE unless severe
            consec = _inc_consecutive(state, action)
            severe = bool(
                (snap.get("action_js") is not None and snap.get("action_js") >= THRESH_ACTION_JS_CRIT)
                or (snap.get("n_feat_psi_hi") is not None and snap.get("n_feat_psi_hi") >= THRESH_NUM_FEATURES_PSI_CRIT)
            )

            if not severe and consec < MIN_CONSECUTIVE:
                state["last_action"] = f"PENDING_{action}"
                state["last_reasons"] = reasons
                save_state(state)
                log.info(f"[agent] action pending: consec={consec}/{MIN_CONSECUTIVE}")
                time.sleep(POLL_SECONDS)
                continue

            # Cooldown check
            if in_cooldown(state):
                state["last_action"] = f"SUPPRESSED_{action}"
                state["last_reasons"] = reasons
                save_state(state)
                log.warning(f"[agent] {action} suppressed (cooldown). reasons={reasons}")
                time.sleep(POLL_SECONDS)
                continue

            plan = {
                "action": action,
                "scope": scope,
                "severity": "critical" if severe else "normal",
                "reasons": reasons,
            }

            audit = llm_audit_note(plan, snap)
            if audit is None:
                audit = deterministic_audit_note(plan, snap)

            # -----------------------
            # Act
            # -----------------------
            conf = {
                "agent_action": action,
                "scope": scope,
                "reasons": reasons,
                "audit_note": audit,   # <-- ADD THIS
                "debug": debug,
                "source_eval_dir": latest_dir_str,
                "timestamp": now_utc_iso(),
            }

            resp = trigger_airflow_dag(conf)

            state["last_triggered_at"] = now_utc_iso()
            state["last_action"] = action
            state["last_reasons"] = reasons
            save_state(state)

            log.info(
                f"[agent] Airflow triggered. dag_id={AIRFLOW_DAG_ID} run={resp.get('dag_run_id')} action={action}"
            )

        except Exception as e:
            log.exception(f"[agent] error: {e}")

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
