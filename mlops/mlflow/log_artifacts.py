# src/mlops/mlflow/log_artifacts.py
from __future__ import annotations
from pathlib import Path
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt

def save_confusion_png_binary(tp: int, tn: int, fp: int, fn: int, out_png: str, title: str):
    """
    Matrix in sklearn-style layout:
        [[tn, fp],
         [fn, tp]]
    """
    cm = np.array([[tn, fp], [fn, tp]], dtype=np.int64)
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)

def save_mag_scatter_png(y_true_pp, y_pred_pp, out_png: str, title: str):
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    plt.scatter(y_true_pp, y_pred_pp, s=6)
    plt.xlabel("True magnitude (pp)")
    plt.ylabel("Pred magnitude (pp)")
    plt.title(title)
    plt.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
