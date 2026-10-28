"""Pure frame identities; hashes establish identity, not fitting/source approval."""
from __future__ import annotations

import hashlib

import pandas as pd

from src.domain.market import TrainingFrameIdentity


def frame_hash(frame) -> str:
    """Byte-compatible with the existing public walk_forward.frame_hash."""
    return hashlib.sha256(
        pd.util.hash_pandas_object(frame, index=True).values.tobytes()
        + str(getattr(frame, "columns", getattr(frame, "name", None))).encode()
    ).hexdigest()


def training_frame_identity(features, labels, feature_available_at, label_available_at, label_end_at):
    """Hash exactly the five supplied aligned frames, with no filtering/restamping."""
    values = dict(features=features, labels=labels, feature_available_at=feature_available_at,
                  label_available_at=label_available_at, label_end_at=label_end_at)
    if not isinstance(features, pd.DataFrame) or features.empty or not features.index.is_unique:
        raise ValueError("nonempty unique indexed feature frame required")
    if any(not isinstance(v, pd.Series) or not v.index.equals(features.index) for v in list(values.values())[1:]):
        raise ValueError("all five frames must have exact aligned identities")
    return TrainingFrameIdentity(**{name: frame_hash(value) for name, value in values.items()})
