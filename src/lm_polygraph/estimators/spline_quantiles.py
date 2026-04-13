import numpy as np

from typing import Dict, List

from .estimator import Estimator


class SplineQuantile(Estimator):
    """
    Token-level q10 uncertainty from spline gate geometry.

    For a chosen layer l and token t, compute:
        d_{t,k} = |h_{t,k}| / ||w_{k}||_2
    and return the 10th quantile over k.

    Important:
    - raw q10 is a *distance-like* quantity
    - smaller q10 => closer to nearby boundaries => more uncertain

    By default this estimator returns NEGATIVE q10 so that
    larger output values correspond to larger uncertainty,
    matching LM-Polygraph's usual convention.
    """

    def __init__(
        self,
        layer_idx: int = -1,
        quantile: float = 0.10,
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
        self.quantile = quantile
        self.eps = eps
        self.return_negative = return_negative

    def __str__(self):
        return f"SplineQ{self.quantile}_layer{self.layer_idx}"


    def __call__(self, stats: Dict[str, np.ndarray]) -> List[np.ndarray]:
        """
        Parameters
        ----------
        stats:
            Dictionary containing:
            - spline_gate_preactivations:
                List[sample][layer] -> np.ndarray of shape [T_i, D_ff]
            - spline_gate_weight_norms:
                List[layer] -> np.ndarray of shape [D_ff]

        Returns
        -------
        List[np.ndarray]
            Token-level uncertainty scores.
            Outer list is batch/sample index.
            Each inner np.ndarray has shape [T_i].
        """
        gate_preactivations = stats["spline_gate_preactivations"]
        gate_weight_norms = stats["spline_gate_weight_norms"]

        out: List[np.ndarray] = []

        for sample_layers in gate_preactivations:
            layer_h = sample_layers[self.layer_idx]  # [T_i, D_ff]
            norms = gate_weight_norms[self.layer_idx]  # [D_ff]

            # Safety against zeros, though norms should already be clamped in caqulator
            norms = np.maximum(norms, self.eps)

            # d_{t,k} = |h_{t,k}| / ||w_k||_2
            d = np.abs(layer_h) / norms[None, :]

            # q10 across neurons/hyperplanes for each token
            q = np.quantile(d, self.quantile, axis=1)

            if self.return_negative:
                q = -q

            out.append(q.astype(np.float32))

        return out


class SplineQuantileSequence(Estimator):
    """
    Sequence-level q10 uncertainty from spline gate geometry.

    For a chosen layer l and token t, compute:
        d_{t,k} = |h_{t,k}| / ||w_{k}||_2
    and return the 10th quantile over k.

    Important:
    - raw q10 is a *distance-like* quantity
    - smaller q10 => closer to nearby boundaries => more uncertain

    By default this estimator returns NEGATIVE q10 so that
    larger output values correspond to larger uncertainty,
    matching LM-Polygraph's usual convention.
    """

    def __init__(
        self,
        layer_idx: int = -1,
        quantile: float = 0.10,
        eps: float = 1e-12,
        agg: str = "mean",
        return_negative: bool = True,
    ):
        super().__init__(
            [
                "spline_gate_preactivations",
                "spline_gate_weight_norms",
            ],
            "sequence",
        )
        self.layer_idx = layer_idx
        self.quantile = quantile
        self.eps = eps
        self.agg = agg
        self.return_negative = return_negative

    def __str__(self):
        return f"SplineQ{self.quantile}_{self.agg}_layer{self.layer_idx}"

    def __call__(self, stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Parameters
        ----------
        stats:
            Dictionary containing:
            - spline_gate_preactivations:
                List[sample][layer] -> np.ndarray of shape [T_i, D_ff]
            - spline_gate_weight_norms:
                List[layer] -> np.ndarray of shape [D_ff]

        Returns
        -------
        np.ndarry of shape batch
        """
        gate_preactivations = stats["spline_gate_preactivations"]
        gate_weight_norms = stats["spline_gate_weight_norms"]

        out = []

        for sample_layers in gate_preactivations:
            layer_h = sample_layers[self.layer_idx]  # [T_i, D_ff]
            norms = gate_weight_norms[self.layer_idx]  # [D_ff]

            # Safety against zeros, though norms should already be clamped in caqulator
            norms = np.maximum(norms, self.eps)

            # d_{t,k} = |h_{t,k}| / ||w_k||_2
            d = np.abs(layer_h) / norms[None, :]

            # q10 across neurons/hyperplanes for each token
            q = np.quantile(d, self.quantile, axis=1)

            if self.return_negative:
                q = -q

            if len(q) == 0:
                seq_score = np.nan
            elif self.agg == "mean":
                seq_score = float(np.mean(q))
            elif self.agg == "max":
                seq_score = float(np.max(q))
            elif self.agg == "min":
                seq_score = float(np.min(q))
            elif self.agg == "std":
                seq_score = float(np.std(q))
            else:
                raise ValueError(f"Unknown aggregation mode: {self.agg}")

            out.append(seq_score)

        return np.array(out, dtype=np.float32)
