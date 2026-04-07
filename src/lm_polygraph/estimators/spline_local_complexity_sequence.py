import numpy as np

from typing import Dict

from .estimator import Estimator


class SplineLocalComplexitySequence(Estimator):
    """
    Sequence-level local complexity at a chosen layer.

    For each token t:
        d_{t,k} = |h_{t,k}| / ||w_k||_2
        lc_t = sum_k 1[d_{t,k} < r]

    Sequence score:
        mean_t(lc_t)

    Larger local complexity => more nearby boundaries => more uncertain.
    """

    def __init__(
        self,
        layer_idx: int = -1,
        r: float = 0.005,
        eps: float = 1e-12,
    ):
        super().__init__(
            [
                "spline_gate_preactivations",
                "spline_gate_weight_norms",
            ],
            "sequence",
        )
        self.layer_idx = layer_idx
        self.r = r
        self.eps = eps

    def __str__(self):
        return f"SplineLocalComplexitySequence_r{self.r}_layer{self.layer_idx}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        gate_preactivations = stats["spline_gate_preactivations"]
        gate_weight_norms = stats["spline_gate_weight_norms"]

        out = []

        for sample_layers in gate_preactivations:
            layer_h = sample_layers[self.layer_idx]   # [T_i, D_ff]
            norms = gate_weight_norms[self.layer_idx] # [D_ff]
            norms = np.maximum(norms, self.eps)

            d = np.abs(layer_h) / norms[None, :]
            lc = np.sum(d < self.r, axis=1)          # [T_i]

            seq_score = float(np.mean(lc)) if len(lc) > 0 else np.nan
            out.append(seq_score)

        return np.array(out, dtype=np.float32)
