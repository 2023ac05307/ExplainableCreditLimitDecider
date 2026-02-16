from __future__ import annotations

import os
import time
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import json
import hashlib
import shutil
from pathlib import Path
import mlflow
import requests
from mlflow.tracking import MlflowClient

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from .model_loader import init_bundle, get_bundle
from .inference import predict_one, InferenceConfig
from .explain import explain_one
from .customer_store import CustomerStore, CustomerStoreConfig
from .schemas import ExplainCustomerResponse, ExplainCustomerItem, ExplainAttribution
from .schemas import ExplainAttribution as SchemaExplainAttribution

from src.serving.metrics import ACTIVE_BUNDLE
from .metrics import (
    track_latency,
    record_request,
    record_prediction,
    metrics_response,
    MODEL_LOAD_SECONDS,
)

from .schemas import (
    PredictRequest, PredictResponse,
    BatchPredictRequest, BatchPredictResponse,
    ExplainRequest, ExplainResponse, FeatureAttribution,
)


def _mlflow_client():
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000").strip()
    if "://" not in tracking_uri:
        tracking_uri = "http://" + tracking_uri
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)

def _get_champion_meta():
    client = _mlflow_client()
    names = {
        "gate": "cl_policy_gate",
        "dir":  "cl_policy_dir",
        "cli":  "cl_mag_cli",
        "cld":  "cl_mag_cld",
    }
    out = {}
    for k, name in names.items():
        try:
            mv = client.get_model_version_by_alias(name, "champion")
            out[k] = {"model": name, "version": mv.version, "run_id": mv.run_id}
        except Exception as e:
            out[k] = {"model": name, "error": str(e)}
    return out


def _sha256_short(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:16]

def _local_ckpt_fingerprints():
    ckpt_root = os.getenv("CKPT_ROOT", "/checkpoints")
    p = Path(ckpt_root)
    files = {
        "gate": p/"classification"/"gate_awac.pt",
        "dir":  p/"classification"/"dir_awac.pt",
        "cli":  p/"regression"/"mag_cli_beta.pt",
        "cld":  p/"regression"/"mag_cld_beta.pt",
    }
    out = {}
    for k, fp in files.items():
        if fp.exists():
            st = fp.stat()
            out[k] = {
                "path": str(fp),
                "size": st.st_size,
                "mtime_epoch": st.st_mtime,
                "mtime_ist": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(st.st_mtime)),
                "sha16": _sha256_short(fp),
            }
        else:
            out[k] = {"path": str(fp), "exists": False}
    return out


def _sync_champion_ckpts_from_registry_by_alias_source(ckpt_root: str, alias: str = "champion") -> dict:
    """
    Downloads champion checkpoint artifacts using model registry alias -> mv.source,
    then automatically overwrites local bind-mounted checkpoint files.

    Returns:
      dict: model_name -> {"version": int, "dst": str, "src": str}
    """


    ckpt_root_p = Path(ckpt_root)
    client = MlflowClient()

    # IMPORTANT: these must match what your trainer registers as artifacts.
    # In your run_all.py you log under checkpoints/classification/... and checkpoints/regression/...
    targets = {
        "cl_policy_gate": ("classification", "gate_awac.pt"),
        "cl_policy_dir":  ("classification", "dir_awac.pt"),
        "cl_mag_cli":     ("regression",     "mag_cli_beta.pt"),
        "cl_mag_cld":     ("regression",     "mag_cld_beta.pt"),
    }

    errors = {}
    pulled = {}

    def _atomic_replace(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)  # atomic swap

    for model_name, (subdir, fname) in targets.items():
        try:
            mv = client.get_model_version_by_alias(model_name, alias)
            src_uri = mv.source
            downloaded_path = Path(mlflow.artifacts.download_artifacts(artifact_uri=src_uri))

            dst = ckpt_root_p / subdir / fname
            _atomic_replace(downloaded_path, dst)

            pulled[model_name] = {
                "version": int(mv.version),
                "src": src_uri,
                "dst": str(dst),
            }
        except Exception as e:
            errors[model_name] = str(e)

    os.environ["MLFLOW_SYNC_ERRORS"] = json.dumps(errors)
    return pulled


# =============================================================================
# UI Request/Response Schemas
# =============================================================================

class CustomerIdsRequest(BaseModel):
    cust_ids: List[str] = Field(..., description="Customer IDs as strings")


class ExplainCustomerIdsRequest(BaseModel):
    cust_ids: List[str]
    stage: str = "auto"   # "gate" | "dir" | "auto"
    top_k: int = 5


class PredictCustomerItem(BaseModel):
    cust_id: str
    next_month: str
    action_taken: str
    magnitude_percentage: float
    prev_credit_limit: float
    updated_credit_limit: float
    gate_prob: float
    dir_prob: Optional[float] = None


class PredictCustomerLimitResponse(BaseModel):
    items: List[PredictCustomerItem]


class ExplainPredictedLimitRequest(BaseModel):
    cust_ids: List[str]
    top_k: int = 5
    stage: str = "auto"
    include_recourse: bool = True
    include_disclosure: bool = True


class ExplainPredictedLimitItem(BaseModel):
    cust_id: str
    next_month: str
    action_taken: str
    magnitude_percentage: float
    prev_credit_limit: float
    updated_credit_limit: float
    gate_prob: float
    dir_prob: Optional[float] = None

    method: str
    stage_used: str
    explanation_lines: List[str]

    customer_explanation: str
    attributions: List[SchemaExplainAttribution]
    recourse: List[str] = Field(default_factory=list)
    disclosure: Dict[str, Any] = Field(default_factory=dict)


class ExplainPredictedLimitResponse(BaseModel):
    items: List[ExplainPredictedLimitItem]


# =============================================================================
# Helpers
# =============================================================================

def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _as_percent_value(mag_pct: float) -> float:
    try:
        v = float(mag_pct)
    except Exception:
        return 0.0
    assume_fraction = os.getenv("MAG_ASSUME_FRACTION", "0").strip() == "1"
    if assume_fraction and (0.0 <= v <= 1.0):
        return v * 100.0
    return v


def _ollama_generate(prompt: str) -> str:
    host = os.getenv("OLLAMA_HOST", "http://ollama:11434").rstrip("/")
    model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    timeout = float(os.getenv("OLLAMA_TIMEOUT", "30"))

    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))},
    }
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return (data.get("response") or "").strip()


def _humanize_feature_name(feat: str) -> str:
    f = feat.lower()
    if "tx" in f or "transaction" in f:
        return "spending / transaction trend"
    if "external_score" in f or "score" in f:
        return "credit score trend"
    if "min_pay" in f or "payment" in f:
        return "payment consistency"
    if "util" in f or "balance" in f:
        return "balance / utilization"
    if "dpd" in f or "delinq" in f:
        return "delinquency indicators"
    if "income" in f:
        return "income stability"
    return "recent account behaviour"


def _sync_ckpts_from_mlflow_registry_best_effort(
    ckpt_root: str,
    alias: str = "champion",
) -> dict:
    """
    Best-effort: tries to download alias-selected artifacts from MLflow registry.
    If one model fails, we skip it (do NOT crash serving).
    Returns details of what was successfully pulled.
    """
    # Ensure scheme is present; MLflow expects http://
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000").strip()
    if "://" not in tracking_uri:
        tracking_uri = "http://" + tracking_uri

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    gate_name    = os.getenv("MLFLOW_MODEL_GATE", "cl_policy_gate")
    dir_name     = os.getenv("MLFLOW_MODEL_DIR",  "cl_policy_dir")
    mag_cli_name = os.getenv("MLFLOW_MODEL_MAG_CLI", "cl_mag_cli")
    mag_cld_name = os.getenv("MLFLOW_MODEL_MAG_CLD", "cl_mag_cld")

    targets = [
        (gate_name,    Path(ckpt_root) / "classification" / "gate_awac.pt"),
        (dir_name,     Path(ckpt_root) / "classification" / "dir_awac.pt"),
        (mag_cli_name, Path(ckpt_root) / "regression" / "mag_cli_beta.pt"),
        (mag_cld_name, Path(ckpt_root) / "regression" / "mag_cld_beta.pt"),
    ]

    out: dict = {}
    errors: dict = {}

    for model_name, dst in targets:
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            mv = client.get_model_version_by_alias(model_name, alias)
            src_uri = mv.source  # e.g., runs:/<run_id>/artifacts/...
            downloaded_path = mlflow.artifacts.download_artifacts(artifact_uri=src_uri)
            shutil.copyfile(downloaded_path, dst)
            out[model_name] = {"version": int(mv.version), "dst": str(dst), "source": src_uri}
        except Exception as e:
            errors[model_name] = str(e)

    if errors:
        # Keep as env for /health debug + container logs
        os.environ["MLFLOW_SYNC_ERRORS"] = str(errors)

    return out


def _local_ckpts_present(ckpt_root: str) -> Dict[str, bool]:
    need = {
        "gate": Path(ckpt_root) / "classification" / "gate_awac.pt",
        "dir":  Path(ckpt_root) / "classification" / "dir_awac.pt",
        "cli":  Path(ckpt_root) / "regression" / "mag_cli_beta.pt",
        "cld":  Path(ckpt_root) / "regression" / "mag_cld_beta.pt",
    }
    return {k: p.exists() for k, p in need.items()}

def _build_recourse(action: str, top_feats: List[str]) -> List[str]:
    """
    Deterministic recourse tips derived from the presence of feature categories.
    No promises; no protected attributes; no speculation.
    """
    cats = []
    for f in top_feats:
        cats.append(_humanize_feature_name(f))

    # de-dup
    seen = set()
    cats_u = []
    for c in cats:
        if c not in seen:
            cats_u.append(c)
            seen.add(c)

    tips = []
    if action == "CLD":
        tips.append("To improve chances of restoring the limit later, focus on payment regularity and reducing utilization.")
    elif action == "HOLD":
        tips.append("To be considered for an increase later, maintain stable repayments and utilization for the next few cycles.")
    else:
        tips.append("To maintain eligibility for future increases, continue consistent repayments and stable utilization.")

    if any("payment" in c for c in cats_u):
        tips.append("Improve payment consistency by paying at least the minimum due on or before the due date each month.")
    if any("utilization" in c or "balance" in c for c in cats_u):
        tips.append("Reduce outstanding balance to lower utilization over the next 1–2 billing cycles.")
    if any("credit score" in c for c in cats_u):
        tips.append("Keep credit utilisation moderate and avoid multiple new credit applications in a short period.")
    if any("spending" in c for c in cats_u):
        tips.append("Keep monthly spending steady; large volatility may trigger conservative decisions.")

    # de-dup preserve order
    out = []
    s2 = set()
    for t in tips:
        if t not in s2:
            out.append(t)
            s2.add(t)
    return out[:6]


def _disclosure_block() -> Dict[str, Any]:
    return {
        "policy_note": "This is a research prototype for academic demonstration. Real bank decisions follow internal policy and compliance checks.",
        "data_note": "Inputs are limited to credit/account behaviour indicators available in the feature store used by this prototype.",
        "protected_attributes": "Protected attributes (e.g., religion, caste, gender) are not used as inputs in this prototype.",
        "fairness_note": "A fairness audit should compare outcomes across customer segments (e.g., score bands, utilization bands) using approval rates and error parity.",
        "recourse_note": "Recourse suggestions are guidance only and do not guarantee a future limit change.",
    }


def _build_ollama_prompt(
    action: str,
    prev_limit: float,
    updated_limit: float,
    mag_pct_points: float,
    top_attribs: List[Tuple[str, float, float]],
) -> str:
    factor_lines = []
    for (f, _, _) in top_attribs[:5]:
        factor_lines.append(f"- {_humanize_feature_name(f)}")
    factors_block = "\n".join(factor_lines) if factor_lines else "- recent account behaviour"

    return f"""
You are rewriting an explanation for a bank customer.

You MUST only use the information provided below.
Do NOT add new reasons.

Decision:
- Action: {action}
- Previous limit: {prev_limit:.2f}
- Updated limit: {updated_limit:.2f}
- Magnitude: {mag_pct_points:.3f}%

Customer-relevant factors (top drivers, simplified):
{factors_block}

Rules:
- Write 1–2 short sentences in a professional bank tone.
- Do NOT mention SHAP, models, probabilities, internal scores, or feature codes.
- Do NOT speculate or mention protected attributes.
- Must clearly explain why the credit limit was held/changed.
- Do NOT introduce new reasons beyond the factors listed.

Write the explanation:
""".strip()


# =============================================================================
# App Factory
# =============================================================================

def create_app() -> FastAPI:
    app = FastAPI(
        title="Explainable Credit Limit Decider",
        version="1.0.0",
        description="Serving for 2-step policy (Gate+Dir) + Beta magnitude with explainability.",
    )

    BASE_DIR = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

    store_holder: dict = {"store": None}

    # ---------------------------
    # Startup (LOCAL FIRST)
    # ---------------------------
    @app.on_event("startup")
    def _startup():
        ckpt_root = os.getenv("CKPT_ROOT", "/checkpoints")  # bind-mounted volume
        device = os.getenv("SERVE_DEVICE", "cpu")

        features_parquet = os.getenv("SERVE_FEATURES_PARQUET", "reports/next_month_prediction")
        cust_id_col = os.getenv("SERVE_CUST_ID_COL", "cust_id")
        feature_prefix = os.getenv("SERVE_FEATURE_PREFIX", "s_")

        use_mlflow = os.getenv("SERVE_USE_MLFLOW", "0") == "1"
        alias = os.getenv("MODEL_ALIAS", "champion")

        t0 = time.perf_counter()
        try:
            # 1️⃣ Check what exists locally (for debugging / health)
            present = _local_ckpts_present(ckpt_root)
            os.environ["LOCAL_CKPTS_PRESENT"] = str(present)

            # 2️⃣ Optional: overlay local checkpoints from MLflow (champion only)
            pulled = {}
            if use_mlflow:
                pulled = _sync_champion_ckpts_from_registry_by_alias_source(
                    ckpt_root=ckpt_root,
                    alias=alias,
                )

            # 3️⃣ Load models from LOCAL filesystem (source of truth at runtime)
            init_bundle(ckpt_root=ckpt_root, device=device)

            # 4️⃣ Bookkeeping for metrics / health
            if pulled:
                parts = []
                for k, v in pulled.items():
                    if isinstance(v, dict) and "version" in v:
                        parts.append(f"{k}:v{v['version']}")
                    else:
                        parts.append(str(k))
                os.environ["MODEL_BUNDLE_ID"] = f"{alias}-" + ",".join(parts)
            else:
                os.environ["MODEL_BUNDLE_ID"] = "local"

            ACTIVE_BUNDLE.labels(bundle_id=os.getenv("MODEL_BUNDLE_ID", "unknown")).set(1)

        except Exception as e:
            raise RuntimeError(
                f"Failed to start API\n"
                f"ckpt_root={ckpt_root}\n"
                f"local_present={os.environ.get('LOCAL_CKPTS_PRESENT')}\n"
                f"use_mlflow={use_mlflow}, alias={alias}\n"
                f"mlflow_sync_errors={os.environ.get('MLFLOW_SYNC_ERRORS')}\n"
                f"error={e}"
            ) from e
        finally:
            MODEL_LOAD_SECONDS.set(time.perf_counter() - t0)

        # Feature store init (unchanged)
        store_holder["store"] = CustomerStore(
            CustomerStoreConfig(
                features_parquet=features_parquet,
                cust_id_col=cust_id_col,
                feature_prefix=feature_prefix,
            )
        )


    # ---------------------------
    # Admin: reload local only (safe)
    # ---------------------------
    @app.post("/admin/reload_local")
    def reload_local():
        ckpt_root = os.getenv("CKPT_ROOT", "/checkpoints")
        device = os.getenv("SERVE_DEVICE", "cpu")
        init_bundle(ckpt_root=ckpt_root, device=device)
        os.environ["MODEL_BUNDLE_ID"] = "local"
        ACTIVE_BUNDLE.labels(bundle_id="local").set(1)
        return {"status": "reloaded_local", "ckpt_root": ckpt_root}

    # ---------------------------
    # Admin: reload from MLflow (overlay then load)
    # ---------------------------
    @app.post("/admin/reload_from_mlflow")
    def reload_from_mlflow():
        ckpt_root = os.getenv("CKPT_ROOT", "/checkpoints")
        device = os.getenv("SERVE_DEVICE", "cpu")
        alias = os.getenv("MODEL_ALIAS", "champion")

        pulled = _sync_ckpts_from_mlflow_registry_best_effort(ckpt_root=ckpt_root, alias=alias)
        init_bundle(ckpt_root=ckpt_root, device=device)

        if pulled:
            os.environ["MODEL_BUNDLE_ID"] = f"{alias}-" + ",".join([f"{k}:v{v['version']}" for k, v in pulled.items()])
        else:
            os.environ["MODEL_BUNDLE_ID"] = "local"

        ACTIVE_BUNDLE.labels(bundle_id=os.getenv("MODEL_BUNDLE_ID", "unknown")).set(1)
        return {
            "status": "reloaded_from_mlflow",
            "alias": alias,
            "pulled": pulled,
            "mlflow_sync_errors": os.environ.get("MLFLOW_SYNC_ERRORS"),
        }

    # ---------------------------
    # Middleware: request metrics
    # ---------------------------
    @app.middleware("http")
    async def prom_middleware(request: Request, call_next):
        endpoint = request.url.path
        with track_latency(endpoint):
            response = await call_next(request)
        record_request(endpoint=endpoint, status_code=response.status_code)
        return response

    # ---------------------------
    # UI route
    # ---------------------------
    @app.get("/", response_class=HTMLResponse)
    def ui(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    # ---------------------------
    # Customers list
    # ---------------------------
    @app.get("/customers")
    def customers(limit: int = 5000):
        store: Optional[CustomerStore] = store_holder.get("store")
        if store is None:
            raise HTTPException(status_code=500, detail="Customer feature store not initialized")
        return {"customer_ids": store.list_customers(limit=limit)}

    # ---------------------------
    # Health + Metrics
    # ---------------------------
    # @app.get("/health")
    # def health():
    #     _ = get_bundle()
    #     return {
    #         "status": "ok",
    #         "bundle_id": os.getenv("MODEL_BUNDLE_ID", "unknown"),
    #         "local_ckpts_present": os.getenv("LOCAL_CKPTS_PRESENT", "{}"),
    #         "mlflow_sync_errors": os.getenv("MLFLOW_SYNC_ERRORS", "{}"),
    #     }

    @app.get("/health")
    def health():
        _ = get_bundle()
        return {
            "status": "ok",
            "bundle_id": os.getenv("MODEL_BUNDLE_ID", "unknown"),
            "local_ckpts_present": os.getenv("LOCAL_CKPTS_PRESENT", "{}"),
            "mlflow_sync_errors": os.getenv("MLFLOW_SYNC_ERRORS", "{}"),
            "champion_registry": _get_champion_meta(),          
            "loaded_checkpoint_fingerprint": _local_ckpt_fingerprints(), 
        }



    @app.get("/metrics")
    def metrics():
        return metrics_response()

    # =============================================================================
    # Existing API: /predict, /predict_batch, /explain
    # =============================================================================

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        try:
            bundle = get_bundle()
            prev_limit = float(req.features.get("s_credit_limit", req.features.get("credit_limit", 0.0)))

            action, mag_pct, updated, dir_prob, gate_prob = predict_one(
                bundle=bundle,
                features=req.features,
                prev_credit_limit=prev_limit,
                next_month="T+1",
                cfg=InferenceConfig(),
            )
            record_prediction(action)

            return PredictResponse(
                cust_id=req.cust_id,
                next_month="T+1",
                action_taken=action,  # type: ignore
                magnitude_percentage=_as_percent_value(mag_pct),
                prev_credit_limit=prev_limit,
                updated_credit_limit=float(updated),
                gate_prob=float(gate_prob),
                dir_prob=_safe_float(dir_prob),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/predict_batch", response_model=BatchPredictResponse)
    def predict_batch(req: BatchPredictRequest):
        try:
            bundle = get_bundle()
            out = []
            for item in req.items:
                prev_limit = float(item.features.get("s_credit_limit", item.features.get("credit_limit", 0.0)))
                action, mag_pct, updated, dir_prob, gate_prob = predict_one(
                    bundle=bundle,
                    features=item.features,
                    prev_credit_limit=prev_limit,
                    next_month="T+1",
                    cfg=InferenceConfig(),
                )
                record_prediction(action)

                out.append(PredictResponse(
                    cust_id=item.cust_id,
                    next_month="T+1",
                    action_taken=action,  # type: ignore
                    magnitude_percentage=_as_percent_value(mag_pct),
                    prev_credit_limit=prev_limit,
                    updated_credit_limit=float(updated),
                    gate_prob=float(gate_prob),
                    dir_prob=_safe_float(dir_prob),
                ))
            return BatchPredictResponse(items=out)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/explain", response_model=ExplainResponse)
    def explain(req: ExplainRequest):
        try:
            bundle = get_bundle()

            prev_limit = float(req.features.get("s_credit_limit", req.features.get("credit_limit", 0.0)))
            action, mag_pct, updated, dir_prob, gate_prob = predict_one(
                bundle=bundle,
                features=req.features,
                prev_credit_limit=prev_limit,
                next_month="T+1",
                cfg=InferenceConfig(),
            )

            method, top, explanation_text, meta = explain_one(
                bundle=bundle,
                features=req.features,
                stage=req.stage,
                top_k=req.top_k,
            )

            attributions = [FeatureAttribution(feature=f, value=v, attribution=a) for (f, v, a) in top]

            return ExplainResponse(
                cust_id=req.cust_id,
                action_taken=action,  # type: ignore
                explanation_lines=explanation_text.splitlines(),
                attributions=attributions,
                method=method,
                meta={
                    **meta,
                    "gate_prob": float(gate_prob),
                    "dir_prob": _safe_float(dir_prob),
                    "magnitude_pct": _as_percent_value(mag_pct),
                    "prev_credit_limit": prev_limit,
                    "updated_limit": float(updated),
                },
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    # =============================================================================
    # UI-driven endpoints
    # =============================================================================

    @app.post("/predict_customer_limit", response_model=PredictCustomerLimitResponse)
    def predict_customer_limit(req: CustomerIdsRequest):
        try:
            bundle = get_bundle()
            store: Optional[CustomerStore] = store_holder.get("store")
            if store is None:
                raise HTTPException(status_code=500, detail="Customer feature store not initialized")

            items: List[PredictCustomerItem] = []
            for cid in req.cust_ids:
                features = store.get_features(str(cid))
                if features is None:
                    continue

                prev_limit = float(features.get("s_credit_limit", features.get("credit_limit", 0.0)))

                action, mag_pct, updated, dir_prob, gate_prob = predict_one(
                    bundle=bundle,
                    features=features,
                    prev_credit_limit=prev_limit,
                    next_month="T+1",
                    cfg=InferenceConfig(),
                )
                record_prediction(action)

                items.append(PredictCustomerItem(
                    cust_id=str(cid),
                    next_month="T+1",
                    action_taken=action,
                    magnitude_percentage=_as_percent_value(mag_pct),
                    prev_credit_limit=float(prev_limit),
                    updated_credit_limit=float(updated),
                    gate_prob=float(gate_prob),
                    dir_prob=_safe_float(dir_prob),
                ))

            return PredictCustomerLimitResponse(items=items)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


    @app.post("/explain_predicted_limit", response_model=ExplainPredictedLimitResponse)
    def explain_predicted_limit(req: ExplainPredictedLimitRequest):
        try:
            bundle = get_bundle()
            store: Optional[CustomerStore] = store_holder.get("store")
            if store is None:
                raise HTTPException(status_code=500, detail="Customer feature store not initialized")

            items: List[ExplainPredictedLimitItem] = []
            ollama_enabled = False
            ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

            for cid in req.cust_ids:
                features = store.get_features(str(cid))
                if features is None:
                    continue

                prev_limit = float(features.get("s_credit_limit", features.get("credit_limit", 0.0)))

                action, mag_pct, updated, dir_prob, gate_prob = predict_one(
                    bundle=bundle,
                    features=features,
                    prev_credit_limit=prev_limit,
                    next_month="T+1",
                    cfg=InferenceConfig(),
                )

                stage_used = req.stage if req.stage != "auto" else ("gate" if action == "HOLD" else "dir")

                method, top, explanation_text, meta = explain_one(
                    bundle=bundle,
                    features=features,
                    stage=stage_used,
                    top_k=req.top_k,
                )

                mag_points = _as_percent_value(mag_pct)

                # Customer-friendly explanation (LLM if available, else template)
                top_feats = [f for (f, _, _) in top]
                customer_expl = ""
                try:
                    prompt = _build_ollama_prompt(
                        action=action,
                        prev_limit=prev_limit,
                        updated_limit=float(updated),
                        mag_pct_points=mag_points,
                        top_attribs=top,
                    )
                    customer_expl = _ollama_generate(prompt)
                    if customer_expl:
                        ollama_enabled = True
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

                recourse = _build_recourse(action=action, top_feats=top_feats) if req.include_recourse else []
                disclosure = _disclosure_block() if req.include_disclosure else {}

                items.append(ExplainPredictedLimitItem(
                    cust_id=str(cid),
                    next_month="T+1",
                    action_taken=action,
                    magnitude_percentage=mag_points,
                    prev_credit_limit=float(prev_limit),
                    updated_credit_limit=float(updated),
                    gate_prob=float(gate_prob),
                    dir_prob=_safe_float(dir_prob),
                    method=method,
                    stage_used=stage_used,
                    explanation_lines=explanation_text.splitlines(),
                    attributions=[SchemaExplainAttribution(feature=f, value=float(v), attribution=float(a)) for (f, v, a) in top],
                    customer_explanation=customer_expl,
                    recourse=recourse,
                    disclosure=disclosure,
                ))

            return {
                "items": items,
                "ollama": {"enabled": ollama_enabled, "model": ollama_model},
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))


    # =============================================================================
    # NEW: Explain predicted limit (LLM rewrite + recourse + disclosure)
    # =============================================================================

    @app.post("/explain_customer_limit", response_model=ExplainCustomerResponse)
    def explain_customer_limit(req: ExplainCustomerIdsRequest):

      try:    
        bundle = get_bundle()
        store: Optional[CustomerStore] = store_holder.get("store")
        if store is None:
            raise HTTPException(status_code=500, detail="Customer feature store not initialized")

        out_items: List[ExplainCustomerItem] = []

        for cid in req.cust_ids:
            features = store.get_features(str(cid))
            if features is None:
                continue

            prev_limit = float(features.get("s_credit_limit", 0.0))

            action, mag_pct, updated, dir_prob, gate_prob = predict_one(
                bundle=bundle,
                features=features,
                prev_credit_limit=prev_limit,
                next_month="T+1",
                cfg=InferenceConfig(),
            )

            stage = req.stage if req.stage != "auto" else ("gate" if action == "HOLD" else "dir")

            method, top, explanation_text, meta = explain_one(
                bundle=bundle,
                features=features,
                stage=stage,
                top_k=req.top_k,
            )

            out_items.append(
                ExplainCustomerItem(
                    cust_id=str(cid),
                    action_taken=action,
                    method=method,
                    explanation_lines=explanation_text.splitlines(),
                    meta={
                        **meta,
                        "gate_prob": float(gate_prob) if gate_prob is not None else None,
                        "dir_prob": float(dir_prob) if dir_prob is not None else None,
                        "magnitude_pct": float(mag_pct),
                        "prev_credit_limit": prev_limit,
                        "updated_limit": float(updated),
                        "stage_used": stage,
                    },
                    attributions=[
                        ExplainAttribution(feature=f, value=float(v), attribution=float(a))
                        for (f, v, a) in top
                    ],
                )
            )

        return ExplainCustomerResponse(items=out_items)

      except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


app = create_app()