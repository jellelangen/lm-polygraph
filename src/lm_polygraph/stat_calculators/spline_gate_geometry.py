import torch
import numpy as np

from typing import Dict, List, Tuple

from .stat_calculator import StatCalculator
from lm_polygraph.utils.model import WhiteboxModel


class SplineGateGeometryCalculator(StatCalculator):
    """
    Collects raw gated-MLP geometry statistics during autoregressive generation.

    Outputs
    -------
    spline_gate_preactivations:
        List[List[np.ndarray]]
        Outer list is batch index i.
        Inner list is layer index l.
        Each array has shape [T_i, D_ff_l], where T_i is the number of generated
        tokens kept for sample i (optionally excluding EOS). Row t is aligned to
        the prediction of generated token t.

    spline_gate_weights:
        List[np.ndarray]
        One array per layer, shape [D_ff_l, D_model_l].

    spline_gate_weight_norms:
        List[np.ndarray]
        One array per layer, shape [D_ff_l].

    spline_generation_token_spans:
        List[Tuple[int, int]]
        For completeness/debugging. Each tuple is (0, T_i), because
        spline_gate_preactivations[i][l] is already sliced to generated-token
        prediction positions only.
    """

    @staticmethod
    def meta_info() -> Tuple[List[str], List[str]]:
        return (
            [
                "spline_gate_preactivations",
                "spline_gate_weights",
                "spline_gate_weight_norms",
                "spline_generation_token_spans",
            ],
            [],
        )

    def __init__(
        self,
        include_eos: bool = False,
        eps: float = 1e-12,
    ):
        super().__init__()
        self.include_eos = include_eos
        self.eps = eps

    def _find_gate_projections(self, hf_model) -> List[Tuple[str, torch.nn.Module]]:
        """
        Finds transformer blocks whose MLP exposes `.gate_proj`.
        This is intentionally generic: it looks for modules with `module.mlp.gate_proj`.
        """
        gate_modules: List[Tuple[str, torch.nn.Module]] = []

        for name, module in hf_model.named_modules():
            mlp = getattr(module, "mlp", None)
            gate_proj = getattr(mlp, "gate_proj", None) if mlp is not None else None

            if gate_proj is not None and hasattr(gate_proj, "weight"):
                gate_modules.append((name, gate_proj))

        if len(gate_modules) == 0:
            raise ValueError(
                "Could not find any gated MLP projections (`module.mlp.gate_proj`). "
                "This calculator currently expects a LLaMA/Mistral/Qwen-style gated MLP."
            )

        return gate_modules

    def _make_hook(self, layer_idx: int, cache: List[List[torch.Tensor]]):
        def hook(module, inp, out):
            # out is typically [B, S, D_ff]
            cache[layer_idx].append(out.detach().cpu())
            return None

        return hook

    def _infer_prompt_lengths(
        self,
        batch: Dict[str, torch.Tensor],
        model: WhiteboxModel,
    ) -> List[int]:
        if "attention_mask" in batch:
            return batch["attention_mask"].sum(dim=1).tolist()

        # Fallback if attention_mask is absent
        input_ids = batch["input_ids"]
        pad_id = model.tokenizer.pad_token_id
        if pad_id is None:
            # No pad token: assume full width is real prompt
            return [input_ids.shape[1]] * input_ids.shape[0]

        return (input_ids != pad_id).sum(dim=1).tolist()

    def _extract_generated_tokens(
        self,
        sequences: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        model: WhiteboxModel,
    ) -> List[List[int]]:
        """
        Matches GreedyProbsCalculator-style slicing for CausalLM:
        generated part is everything after the prompt width.
        """
        if model.model_type != "CausalLM":
            raise NotImplementedError(
                "SplineGateGeometryCalculator currently supports CausalLM only."
            )

        prompt_width = batch["input_ids"].shape[1]
        seqs: List[List[int]] = []

        for i in range(sequences.shape[0]):
            seq = sequences[i, prompt_width:].cpu().tolist()

            if model.tokenizer.eos_token_id is not None and model.tokenizer.eos_token_id in seq:
                eos_idx = seq.index(model.tokenizer.eos_token_id)
                if self.include_eos:
                    seq = seq[: eos_idx + 1]
                else:
                    seq = seq[:eos_idx]

            seqs.append(seq)

        return seqs

    def __call__(
        self,
        dependencies: Dict[str, np.ndarray],
        texts: List[str],
        model: WhiteboxModel,
        max_new_tokens: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Runs generation once with forward hooks on all gate projections.

        Alignment convention
        --------------------
        For generated token y_t, we store the gate preactivation vector from the
        hidden state that *predicts* y_t.

        That means:
        - y_1 is aligned to the final real prompt position from the initial prompt pass
        - y_2 is aligned to the first decode-step hook output
        - y_3 is aligned to the second decode-step hook output
        - etc.

        This aligns rows with generated-token prediction events rather than with
        token embeddings after they are already appended to the context.
        """
        if not isinstance(model, WhiteboxModel):
            raise TypeError(
                "SplineGateGeometryCalculator currently supports lm_polygraph.utils.model.WhiteboxModel only."
            )

        hf_model = model.model
        gate_modules = self._find_gate_projections(hf_model)

        # One list per layer; each element in that list is one forward-call chunk
        layer_cache: List[List[torch.Tensor]] = [[] for _ in range(len(gate_modules))]
        handles = []

        try:
            for layer_idx, (_, gate_proj) in enumerate(gate_modules):
                h = gate_proj.register_forward_hook(
                    self._make_hook(layer_idx, layer_cache)
                )
                handles.append(h)

            batch: Dict[str, torch.Tensor] = model.tokenize(texts)
            prompt_lengths = self._infer_prompt_lengths(batch, model)
            batch = {k: v.to(model.device()) for k, v in batch.items()}

            with torch.no_grad():
                generation = model.generate(
                    **batch,
                    output_scores=False,
                    return_dict_in_generate=True,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=2,
                    num_return_sequences=1,
                    suppress_tokens=(
                        []
                        if model.generation_parameters.allow_newlines
                        else [
                            t
                            for t in range(len(model.tokenizer))
                            if "\n" in model.tokenizer.decode([t])
                        ]
                    ),
                )

            sequences = generation.sequences
            generated_tokens = self._extract_generated_tokens(sequences, batch, model)

        finally:
            for h in handles:
                h.remove()

        # Collect layer-constant weights and norms
        gate_weights: List[np.ndarray] = []
        gate_weight_norms: List[np.ndarray] = []

        for _, gate_proj in gate_modules:
            w = gate_proj.weight.detach().cpu()
            gate_weights.append(w.numpy())
            gate_weight_norms.append(
                w.norm(p=2, dim=1).clamp_min(self.eps).numpy()
            )

        # Build sample -> layer -> [T_i, D_ff] arrays
        # Hook behavior:
        #   layer_cache[l][0] = prompt pass chunk, shape [B, S_prompt_padded, D_ff]
        #   layer_cache[l][1:] = decode chunks, typically each [B, 1, D_ff]
        #
        # We align prediction positions to generated tokens:
        #   token 1 -> last real prompt position
        #   token 2..T -> first T-1 decode steps
        spline_gate_preactivations: List[List[np.ndarray]] = []
        spline_generation_token_spans: List[Tuple[int, int]] = []

        for i in range(len(texts)):
            T_i = len(generated_tokens[i])
            spline_generation_token_spans.append((0, T_i))
            per_sample_layers: List[np.ndarray] = []

            for l in range(len(gate_modules)):
                chunks = layer_cache[l]
                if len(chunks) == 0:
                    raise RuntimeError(
                        f"No hook outputs captured for layer {l}. "
                        "This likely means the gate hooks were not triggered."
                    )

                prompt_chunk = chunks[0]  # [B, S_prompt_padded, D_ff]
                d_ff = prompt_chunk.shape[-1]

                # Final real prompt position predicts the first generated token.
                last_prompt_vec = prompt_chunk[i, -1, :].view(1, d_ff)

                if len(chunks) > 1:
                    decode_cat = torch.cat(chunks[1:], dim=1)  # [B, steps, D_ff]
                else:
                    decode_cat = torch.empty(
                        (prompt_chunk.shape[0], 0, d_ff), dtype=prompt_chunk.dtype
                    )

                if T_i == 0:
                    aligned = torch.empty((0, d_ff), dtype=prompt_chunk.dtype)
                elif T_i == 1:
                    aligned = last_prompt_vec
                else:
                    # Need T_i - 1 decode-step vectors after the first token
                    aligned = torch.cat(
                        [
                            last_prompt_vec,
                            decode_cat[i, : T_i - 1, :],
                        ],
                        dim=0,
                    )

                per_sample_layers.append(aligned.numpy())

            spline_gate_preactivations.append(per_sample_layers)

        return {
            "spline_gate_preactivations": spline_gate_preactivations,
            "spline_gate_weights": gate_weights,
            "spline_gate_weight_norms": gate_weight_norms,
            "spline_generation_token_spans": spline_generation_token_spans,
        }
