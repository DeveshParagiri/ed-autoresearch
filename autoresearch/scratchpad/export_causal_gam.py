"""Export the ecologically filtered causal GAM as a dependency-free equation.

The export retains every one-dimensional response and the 160 strongest
pairwise process couplings after removing interactions involving urban fraction.
Urban cover remains available as a one-dimensional fragmentation response, but
is not allowed to condition climate or lightning as an implicit location proxy.
"""

from __future__ import annotations

import base64
import pickle
import zlib
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "causal_gam.b85"
OUTPUT = ROOT / "causal_gam_compact.b85"


def predict(payload: dict[str, object], values: np.ndarray) -> np.ndarray:
    raw = np.full(values.shape[0], float(payload["intercept"]), dtype=np.float64)
    bins = payload["bins"]
    terms = payload["terms"]
    assert isinstance(bins, tuple) and isinstance(terms, tuple)
    for features, scores in terms:
        level = len(features) - 1
        indices = []
        for feature in features:
            feature_bins = bins[feature]
            cuts = feature_bins[min(level, len(feature_bins) - 1)]
            indices.append(
                np.searchsorted(cuts, values[:, feature], side="right") + 1
            )
        raw += scores[tuple(indices)]
    return np.exp(np.clip(raw, -30.0, 30.0))


def main() -> int:
    learner = pickle.loads(
        zlib.decompress(base64.b85decode(SOURCE.read_bytes()))
    )
    importance = learner.term_importances()
    main_terms = [
        index
        for index, features in enumerate(learner.term_features_)
        if len(features) == 1
    ]
    interactions = sorted(
        (
            index
            for index, features in enumerate(learner.term_features_)
            if len(features) == 2
            and "luh2_urban_fraction" not in learner.term_names_[index]
        ),
        key=lambda index: importance[index],
        reverse=True,
    )[:160]
    retained = set(main_terms + interactions)
    payload = {
        "feature_names": tuple(str(name) for name in learner.feature_names_in_),
        "intercept": float(np.asarray(learner.intercept_).reshape(-1)[0]),
        "bins": tuple(
            tuple(np.asarray(cuts, dtype=np.float64) for cuts in feature_bins)
            for feature_bins in learner.bins_
        ),
        "terms": tuple(
            (
                tuple(int(feature) for feature in features),
                np.asarray(scores, dtype=np.float64),
            )
            for index, (features, scores) in enumerate(
                zip(learner.term_features_, learner.term_scores_, strict=True)
            )
            if index in retained
        ),
    }
    encoded = base64.b85encode(
        zlib.compress(pickle.dumps(payload, protocol=5), level=9)
    )
    OUTPUT.write_bytes(encoded)

    rng = np.random.default_rng(192)
    test = np.empty((2000, len(payload["feature_names"])), dtype=np.float64)
    for index, feature_bins in enumerate(payload["bins"]):
        cuts = feature_bins[0]
        test[:, index] = rng.uniform(cuts[0] - 1.0, cuts[-1] + 1.0, test.shape[0])
    reduced = pickle.loads(pickle.dumps(learner, protocol=5))
    for index, scores in enumerate(reduced.term_scores_):
        if index not in retained:
            scores.fill(0.0)
    expected = reduced.predict(test)
    actual = predict(payload, test)
    print(
        f"features={len(payload['feature_names'])} terms={len(payload['terms'])} "
        f"interactions={len(interactions)} bytes={len(encoded)} "
        f"max_abs_error={np.max(np.abs(expected - actual)):.3e}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
