# src/mlops/mlflow/promote.py
from __future__ import annotations
from typing import Dict, Optional, Tuple
from mlflow.tracking import MlflowClient

def _get_version_tags(client: MlflowClient, model: str, version: str) -> Dict[str, str]:
    mv = client.get_model_version(model, version)
    return dict(mv.tags or {})

def should_promote_classification(
    ch_tags: Dict[str, str], cl_tags: Dict[str, str]
) -> Tuple[bool, str]:
    ch_f1 = float(ch_tags["f1"])
    cl_f1 = float(cl_tags["f1"])

    if cl_f1 > ch_f1:
        return True, "Higher macro-F1"
    if cl_f1 < ch_f1:
        return False, "Lower macro-F1"

    ch_ba = float(ch_tags["balanced_acc"])
    cl_ba = float(cl_tags["balanced_acc"])

    if cl_ba > ch_ba:
        return True, "Tie on F1, higher balanced accuracy"
    return False, "Tie on F1, lower or equal balanced accuracy"

def should_promote_regression(
    ch_tags: Dict[str, str], cl_tags: Dict[str, str]
) -> Tuple[bool, str]:
    ch_mae = float(ch_tags["mae_pp"])
    cl_mae = float(cl_tags["mae_pp"])

    if cl_mae < ch_mae:
        return True, "Lower MAE"
    if cl_mae > ch_mae:
        return False, "Higher MAE"

    ch_rmse = float(ch_tags["rmse_pp"])
    cl_rmse = float(cl_tags["rmse_pp"])

    if cl_rmse < ch_rmse:
        return True, "Tie on MAE, lower RMSE"
    return False, "Tie on MAE, higher or equal RMSE"


def _assert_required_tags(
    client: MlflowClient,
    tags: dict,
    required: list,
    model_name: str,
    version: str,
    role: str,
):
    missing = [k for k in required if k not in tags]

    if not missing:
        return

    if role == "champion":
        for k in missing:
            client.set_model_version_tag(model_name, version, k, "0.0")
        print(
            f"[PROMOTE][WARN] Backfilled missing tags {missing} "
            f"for existing champion {model_name} v{version}"
        )
        return

    raise RuntimeError(
        f"[PROMOTE] Missing required metric tags {missing} "
        f"for model={model_name}, version={version}"
    )

def enforce_single_champion_tag(model_name: str):
    client = MlflowClient()
    champ = client.get_model_version_by_alias(model_name, "champion")

    for mv in client.search_model_versions(f"name='{model_name}'"):
        if str(mv.version) == str(champ.version):
            client.set_model_version_tag(model_name, mv.version, "role", "champion")
        else:
            try:
                client.delete_model_version_tag(model_name, mv.version, "role")
            except Exception:
                pass

def promote_if_better(model_name: str, challenger_version: str, kind: str) -> Tuple[bool, str]:
    """
    kind: 'clf' or 'reg'
    """
    client = MlflowClient()

    # Find current champion
    try:
        champion = client.get_model_version_by_alias(model_name, "champion")
        champion_version = str(champion.version)
    except Exception:
        # no champion yet
        client.set_registered_model_alias(model_name, "champion", challenger_version)
        return True, f"No champion existed. Set v{challenger_version} as champion."

    ch_tags = _get_version_tags(client, model_name, champion_version)
    cl_tags = _get_version_tags(client, model_name, challenger_version)

    if kind == "clf":
        REQUIRED = ["f1", "balanced_acc"]
        _assert_required_tags(client, ch_tags, REQUIRED, model_name, champion_version, role="champion")
        ch_tags = _get_version_tags(client, model_name, champion_version)
        _assert_required_tags(client, cl_tags, REQUIRED, model_name, challenger_version, role="challenger")
        promote, reason = should_promote_classification(ch_tags, cl_tags)
    else:
        REQUIRED = ["mae_pp", "rmse_pp"]
        _assert_required_tags(client, ch_tags, REQUIRED, model_name, champion_version, role="champion")
        ch_tags = _get_version_tags(client, model_name, champion_version)
        _assert_required_tags(client, cl_tags, REQUIRED, model_name, challenger_version, role="challenger")
        promote, reason = should_promote_regression(ch_tags, cl_tags)

    if promote:
        client.set_registered_model_alias(model_name, "champion", challenger_version)
        client.set_model_version_tag(model_name, challenger_version, "promotion_reason", reason)
        return True, f"Promoted v{challenger_version} over v{champion_version}."
    return False, f"Kept champion v{champion_version}; challenger v{challenger_version} not better."
