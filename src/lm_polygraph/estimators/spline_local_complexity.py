import numpy as np

from typing import Dict, List

from .estimator import Estimator


class SplineLocalComplexity(Estimator):
    """
    Token-level local complexity at a chosen layer.

    For each token t:
        d_{t,k} = |h_{t,k}| / ||w_k||_2
        lc_t = sum_k 1[d_{t,k} < r]

    Larger lc => more nearby boundaries => more uncertain.
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
            "token",
        )
        self.layer_idx = layer_idx
        self.r = r
        self.eps = eps

    def __str__(self):
        # Keep r in the name so different radii can coexist in one benchmark
        return f"SplineLocalComplexity_r{self.r}_layer{self.layer_idx}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> List[np.ndarray]:
        gate_preactivations = stats["spline_gate_preactivations"]
        gate_weight_norms = stats["spline_gate_weight_norms"]

        out: List[np.ndarray] = []

        for sample_layers in gate_preactivations:
            layer_h = sample_layers[self.layer_idx]          # [T_i, D_ff]
            norms = gate_weight_norms[self.layer_idx]        # [D_ff]
            norms = np.maximum(norms, self.eps)

            d = np.abs(layer_h) / norms[None, :]
            lc = np.sum(d < self.r, axis=1)

            out.append(lc.astype(np.float32))

        return out


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
        agg: str = "mean",
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
        self.agg = agg
        self.eps = eps

    def __str__(self):
        return f"SplineLocalComplexity_{self.agg}_r{self.r}_layer{self.layer_idx}"

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

            if len(lc) == 0:
                seq_score = np.nan
            elif self.agg == "mean":
                seq_score = float(np.mean(lc))
            elif self.agg == "max":
                seq_score = float(np.max(lc))
            elif self.agg == "min":
                seq_score = float(np.min(lc))
            elif self.agg == "std":
                seq_score = float(np.std(lc))
            else:
                raise ValueError(f"Unknown aggregation mode: {self.agg}")

            out.append(seq_score)

        return np.array(out, dtype=np.float32)
