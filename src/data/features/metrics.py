"""Aligned per-date IC. Undefined statistics stay undefined."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score


def prediction_metrics(predictions: pd.Series, labels: pd.Series) -> dict[str, float]:
    if not predictions.index.equals(labels.index) or not labels.index.is_unique:
        raise ValueError("prediction/label identities must match exactly")
    frame = pd.DataFrame({"p": predictions, "y": labels}).dropna()
    ics = []
    if isinstance(frame.index, pd.MultiIndex):
        date = next((n for n in ("datetime", "date", "timestamp") if n in frame.index.names), None)
        if date is None:
            raise ValueError("date level is required for IC")
        for _, chunk in frame.groupby(level=date):
            if len(chunk) >= 3 and chunk.p.nunique() > 1 and chunk.y.nunique() > 1:
                ics.append(float(spearmanr(chunk.p, chunk.y).statistic))
    elif len(frame) >= 3 and frame.p.nunique() > 1 and frame.y.nunique() > 1:
        # A pooled diagnostic is not a temporal ICIR.
        ics.append(float(spearmanr(frame.p, frame.y).statistic))
    mean = float(np.mean(ics)) if ics else float("nan")
    std = float(np.std(ics, ddof=1)) if len(ics) > 1 else float("nan")
    return {"ic": mean, "icir": mean / std if std > 0 else float("nan"),
            "mse": float(mean_squared_error(frame.y, frame.p)) if len(frame) else float("nan"),
            "r2": float(r2_score(frame.y, frame.p)) if len(frame) > 1 else float("nan")}
