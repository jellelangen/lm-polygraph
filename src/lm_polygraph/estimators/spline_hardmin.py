import numpy as np

from typing import Dict, List

from .estimator import Estimator


class SplineHardMin(Estimator):
    """
    Token-level hard minimum distance to a gate hyperplane at a chosen layer.

    For each token t:
        d_{t,k} = |h_{t,k}| / ||w_k||_2
        hardmin_t = min_k d_{t,k}

    Smaller hardmin => closer to a boundary => more uncertain.
    By default we return NEGATIVE hardmin so that larger output values
    correspond to larger uncertainty.
    """

    def __init__(
        self,
        layer_idx: int = -1,
        eps: float = 1e-12,
        return_negative: bool = True,
    ):
        super().__init__(
            [
                "spline_gate_preactivations",
                "spline_gate_weight_norms",
            ],
            "token",
        )
        self.layer_idx = layer_idx
        self.eps = eps
        self.return_negative = return_negative

    def __str__(self):
        return f"SplineHardMin_layer{self.layer_idx}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> List[np.ndarray]:
        gate_preactivations = stats["spline_gate_preactivations"]
        gate_weight_norms = stats["spline_gate_weight_norms"]

        out: List[np.ndarray] = []

        for sample_layers in gate_preactivations:
            layer_h = sample_layers[self.layer_idx]          # [T_i, D_ff]
            norms = gate_weight_norms[self.layer_idx]        # [D_ff]
            norms = np.maximum(norms, self.eps)

            d = np.abs(layer_h) / norms[None, :]
            hardmin = np.min(d, axis=1)

            if self.return_negative:
                hardmin = -hardmin

            out.append(hardmin.astype(np.float32))

        return out
