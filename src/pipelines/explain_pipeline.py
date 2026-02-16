# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# from __future__ import annotations
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Dict, Any, Optional, List

# import pandas as pd


# def _ensure_dir(p: Path) -> None:
#     p.mkdir(parents=True, exist_ok=True)


# @dataclass
# class ExplainPipelineConfig:
#     features_parquet: str                 # last-month features with s_ columns + cust_id
#     preds_parquet: str                    # from inference/eval pipeline
#     out_dir: str = "reports/explain"

#     cust_ids: Optional[List[int]] = None
#     top_k: int = 8
#     stage: str = "final"                  # gate/dir/final

#     # checkpoints root (serving bundle uses these)
#     ckpt_root: str = "checkpoints"


# def run_explain_pipeline(conf: ExplainPipelineConfig) -> Dict[str, Any]:
#     out = Path(conf.out_dir)
#     _ensure_dir(out)

#     feats = pd.read_parquet(conf.features_parquet)
#     preds = pd.read_parquet(conf.preds_parquet)

#     if conf.cust_ids:
#         feats = feats[feats["cust_id"].isin(conf.cust_ids)].copy()
#         preds = preds[preds["cust_id"].isin(conf.cust_ids)].copy()

#     # join base table
#     base = preds.merge(feats, on="cust_id", how="inner")
#     if base.empty:
#         raise RuntimeError("No matching cust_id between features_parquet and preds_parquet.")

#     # load serving bundle + explainer
#     from src.serving.model_loader import init_bundle
#     from src.serving.explain import explain_one

#     bundle = init_bundle(conf.ckpt_root, device="cpu")

#     feat_cols = [c for c in base.columns if c.startswith("s_")]
#     rows_attr = []
#     rows_text = []

#     for _, r in base.iterrows():
#         cust_id = int(r["cust_id"])
#         feat_map = {c: float(r[c]) for c in feat_cols}

#         method, top, explanation_text, meta = explain_one(
#             bundle=bundle,
#             features=feat_map,
#             stage=conf.stage,
#             top_k=conf.top_k,
#         )

#         # attribution rows
#         for rank, (f, v, a) in enumerate(top, start=1):
#             rows_attr.append({
#                 "cust_id": cust_id,
#                 "stage": conf.stage,
#                 "method": method,
#                 "rank": rank,
#                 "feature": f,
#                 "value": v,
#                 "attribution": a,
#             })

#         rows_text.append({
#             "cust_id": cust_id,
#             "method": method,
#             "stage": conf.stage,
#             "explanation": explanation_text,
#         })

#     attr_df = pd.DataFrame(rows_attr)
#     text_df = pd.DataFrame(rows_text)

#     attr_path = out / "feature_attributions.parquet"
#     text_path = out / "explanations.parquet"
#     html_path = out / "explanations.html"

#     attr_df.to_parquet(attr_path, index=False)
#     text_df.to_parquet(text_path, index=False)

#     # build a clean HTML table for viva/report screenshots
#     merged = preds.merge(text_df[["cust_id", "explanation"]], on="cust_id", how="left")
#     merged.to_html(html_path, index=False)

#     return {
#         "feature_attributions_parquet": str(attr_path),
#         "explanations_parquet": str(text_path),
#         "explanations_html": str(html_path),
#         "n_explained": int(len(text_df)),
#     }


# def main():
#     import argparse
#     p = argparse.ArgumentParser()
#     p.add_argument("--features", required=True)
#     p.add_argument("--preds", required=True)
#     p.add_argument("--out_dir", default="reports/explain")
#     p.add_argument("--top_k", type=int, default=8)
#     p.add_argument("--stage", default="final", choices=["gate", "dir", "final"])
#     args = p.parse_args()

#     conf = ExplainPipelineConfig(
#         features_parquet=args.features,
#         preds_parquet=args.preds,
#         out_dir=args.out_dir,
#         top_k=args.top_k,
#         stage=args.stage,
#     )
#     out = run_explain_pipeline(conf)
#     print("Wrote:", out["explanations_html"])


# if __name__ == "__main__":
#     main()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class ExplainPipelineConfig:
    features_parquet: str
    preds_parquet: str
    out_dir: str = "reports/explain"

    cust_ids: Optional[List[int]] = None
    top_k: int = 8
    stage: str = "final"                  # gate/dir/final

    ckpt_root: str = "checkpoints"

    # ---- speed / runtime knobs (like training scripts) ----
    device: str = "cpu"                   # cpu/cuda
    batch_size: int = 256                 # batching rows through the loop
    num_workers: int = 0                  # threads for explanation (0=off)
    pin_memory: bool = False              # relevant for cuda tensor transfers
    non_blocking: bool = True             # relevant for cuda tensor transfers
    limit_rows: Optional[int] = None      # quick debug: explain only first N rows


def run_explain_pipeline(conf: ExplainPipelineConfig) -> Dict[str, Any]:
    out = Path(conf.out_dir)
    _ensure_dir(out)

    # Load only needed columns to speed IO
    feats = pd.read_parquet(conf.features_parquet)
    preds = pd.read_parquet(conf.preds_parquet)

    if conf.cust_ids:
        feats = feats[feats["cust_id"].isin(conf.cust_ids)].copy()
        preds = preds[preds["cust_id"].isin(conf.cust_ids)].copy()

    # Join base table
    base = preds.merge(feats, on="cust_id", how="inner")
    if base.empty:
        raise RuntimeError("No matching cust_id between features_parquet and preds_parquet.")

    # Optional row limit (debug / quick runs)
    if conf.limit_rows is not None:
        base = base.head(int(conf.limit_rows)).copy()

    # load serving bundle + explainer
    from src.serving.model_loader import init_bundle
    from src.serving.explain import explain_one

    bundle = init_bundle(conf.ckpt_root, device=conf.device)

    # Use only s_ columns (fast)
    feat_cols = [c for c in base.columns if c.startswith("s_")]
    if not feat_cols:
        raise RuntimeError("No s_ feature columns found after join. Check features_parquet content.")

    # -------- Fast row access: itertuples (much faster than iterrows) --------
    # We’ll iterate in batches to reduce Python overhead.
    rows_attr = []
    rows_text = []

    # Threaded parallelism (optional). Safer than multiprocessing because bundle/models stay in-memory.
    use_threads = conf.num_workers and conf.num_workers > 0
    if use_threads:
        from concurrent.futures import ThreadPoolExecutor

        def _explain_row(cust_id: int, feat_map: Dict[str, float]):
            return cust_id, explain_one(
                bundle=bundle,
                features=feat_map,
                stage=conf.stage,
                top_k=conf.top_k,
            )

    # Build a lightweight accessor mapping from tuple -> features dict
    # (Still makes a dict per row, but avoids Pandas overhead and is much faster overall.)
    tuples = list(base[["cust_id"] + feat_cols].itertuples(index=False, name=None))

    def _process_one(cust_id: int, feat_values: tuple):
        feat_map = {c: float(v) for c, v in zip(feat_cols, feat_values)}
        return cust_id, explain_one(
            bundle=bundle,
            features=feat_map,
            stage=conf.stage,
            top_k=conf.top_k,
        )

    # Batch loop
    bs = max(1, int(conf.batch_size))
    for start in range(0, len(tuples), bs):
        chunk = tuples[start:start + bs]

        if use_threads:
            with ThreadPoolExecutor(max_workers=int(conf.num_workers)) as ex:
                futs = []
                for row in chunk:
                    cust_id = int(row[0])
                    feat_values = row[1:]
                    feat_map = {c: float(v) for c, v in zip(feat_cols, feat_values)}
                    futs.append(ex.submit(_explain_row, cust_id, feat_map))

                for f in futs:
                    cust_id, (method, top, explanation_text, meta) = f.result()

                    for rank, (fname, fval, attr) in enumerate(top, start=1):
                        rows_attr.append({
                            "cust_id": cust_id,
                            "stage": conf.stage,
                            "method": method,
                            "rank": rank,
                            "feature": fname,
                            "value": fval,
                            "attribution": attr,
                        })

                    rows_text.append({
                        "cust_id": cust_id,
                        "method": method,
                        "stage": conf.stage,
                        "explanation": explanation_text,
                    })
        else:
            for row in chunk:
                cust_id = int(row[0])
                feat_values = row[1:]

                cust_id, (method, top, explanation_text, meta) = _process_one(cust_id, feat_values)

                for rank, (fname, fval, attr) in enumerate(top, start=1):
                    rows_attr.append({
                        "cust_id": cust_id,
                        "stage": conf.stage,
                        "method": method,
                        "rank": rank,
                        "feature": fname,
                        "value": fval,
                        "attribution": attr,
                    })

                rows_text.append({
                    "cust_id": cust_id,
                    "method": method,
                    "stage": conf.stage,
                    "explanation": explanation_text,
                })

    # Write outputs
    attr_df = pd.DataFrame(rows_attr)
    text_df = pd.DataFrame(rows_text)

    attr_path = out / "feature_attributions.parquet"
    text_path = out / "explanations.parquet"
    html_path = out / "explanations.html"

    attr_df.to_parquet(attr_path, index=False)
    text_df.to_parquet(text_path, index=False)

    merged = preds.merge(text_df[["cust_id", "explanation"]], on="cust_id", how="left")
    merged.to_html(html_path, index=False)

    return {
        "feature_attributions_parquet": str(attr_path),
        "explanations_parquet": str(text_path),
        "explanations_html": str(html_path),
        "n_explained": int(len(text_df)),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--preds", required=True)
    p.add_argument("--out_dir", default="reports/explain")
    p.add_argument("--top_k", type=int, default=8)
    p.add_argument("--stage", default="final", choices=["gate", "dir", "final"])

    # speed knobs
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0, help="Thread workers for explain() calls (0=off)")
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--non_blocking", action="store_true")
    p.add_argument("--limit_rows", type=int, default=None, help="Explain only first N rows (debug)")

    args = p.parse_args()

    conf = ExplainPipelineConfig(
        features_parquet=args.features,
        preds_parquet=args.preds,
        out_dir=args.out_dir,
        top_k=args.top_k,
        stage=args.stage,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        non_blocking=args.non_blocking,
        limit_rows=args.limit_rows,
    )
    out = run_explain_pipeline(conf)
    print("Wrote:", out["explanations_html"])


if __name__ == "__main__":
    main()
