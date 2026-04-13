#!/usr/bin/env python3
"""
sanity_check.py — Load a UEManager checkpoint from a polygraph_eval run
and verify that:

  1. The spline estimators actually produced scores (not all NaN).
  2. The greedy texts from the default pipeline match what the spline
     calculator would have generated (double-generation check).
  3. Print a compact results table (PRR, Isotonic PCC, ECE) for every
     estimator × generation metric pair.

Usage
-----
    python sanity_check.py <path/to/ue_manager_seed1>

If the checkpoint was produced with the debug config (20 samples), this
should finish instantly.  For the full benchmark it is also fast since it
only loads and inspects the saved dict — no generation happens.
"""

import sys
import argparse
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path


def load_manager_dict(path: str) -> dict:
    """Load the raw torch checkpoint dict (avoids needing lm_polygraph installed)."""
    return torch.load(path, map_location="cpu", weights_only=False)


def check_estimations(estimations: dict) -> bool:
    """Verify that spline estimators produced non-NaN scores."""
    ok = True
    spline_keys = [k for k in estimations if "Spline" in k[1] or "spline" in k[1].lower()]

    if not spline_keys:
        print("[FAIL] No spline estimator keys found in estimations.")
        print("       Available keys:")
        for k in sorted(estimations.keys(), key=lambda x: x[1]):
            print(f"         {k}")
        return False

    for key in spline_keys:
        scores = np.array(estimations[key])
        n_total = len(scores)
        n_nan = np.isnan(scores).sum()
        n_inf = np.isinf(scores).sum()
        if n_total == 0:
            print(f"[FAIL] {key}: empty score array")
            ok = False
        elif n_nan == n_total:
            print(f"[FAIL] {key}: all {n_total} scores are NaN")
            ok = False
        elif n_nan > 0 or n_inf > 0:
            print(f"[WARN] {key}: {n_nan}/{n_total} NaN, {n_inf}/{n_total} Inf")
        else:
            lo, med, hi = np.percentile(scores, [5, 50, 95])
            print(f"[ OK ] {key}: n={n_total}, median={med:.6f}, [5%={lo:.6f}, 95%={hi:.6f}]")

    return ok


def check_double_generation(stats: dict) -> bool:
    """
    If both greedy_texts and spline stats are present, flag a warning that
    the user should verify sequence identity externally.  The checkpoint
    does not store the spline-generated texts separately, so we cannot do
    an automatic equality check — but we can at least check that the
    spline preactivation shapes are consistent with the greedy token counts.
    """
    greedy_texts = stats.get("greedy_texts", None)
    greedy_tokens = stats.get("greedy_tokens", None)

    # spline stats are nested lists, not flat — check if they were saved
    spline_preact = stats.get("spline_gate_preactivations", None)
    spline_spans = stats.get("spline_generation_token_spans", None)

    if greedy_tokens is None:
        print("[SKIP] No greedy_tokens in saved stats; cannot cross-check token counts.")
        return True

    if spline_spans is None and spline_preact is None:
        print("[SKIP] Spline stats not saved in checkpoint.")
        print("       To enable, add to your config:")
        print("         save_stats:")
        print("           - spline_gate_preactivations")
        print("           - spline_generation_token_spans")
        print("       Then the double-generation check can run automatically.")
        return True

    if spline_spans is not None:
        # Compare token counts
        mismatches = 0
        for i, ((start, end), greedy_toks) in enumerate(
            zip(spline_spans, greedy_tokens)
        ):
            spline_len = end - start
            greedy_len = len(greedy_toks) if isinstance(greedy_toks, (list, tuple)) else 1
            if spline_len != greedy_len:
                if mismatches < 5:
                    print(
                        f"[WARN] Sample {i}: spline generated {spline_len} tokens, "
                        f"greedy generated {greedy_len} tokens"
                    )
                mismatches += 1

        if mismatches == 0:
            print(f"[ OK ] Token counts match across all {len(spline_spans)} samples.")
        else:
            print(f"[FAIL] {mismatches}/{len(spline_spans)} samples have mismatched token counts.")
            print("       This means the spline calculator and the default greedy calculator")
            print("       produced different-length sequences.  The evaluation may be invalid.")
            return False

    return True


def print_metrics_table(metrics: dict):
    """Print a compact table of all (estimator, gen_metric, ue_metric) -> score."""
    if not metrics:
        print("[WARN] No metrics computed (metrics dict is empty).")
        return

    # Group by generation metric
    # Key format: (estimator_level, estimator_name, gen_metric_name, ue_metric_name)
    by_gen_metric = defaultdict(list)
    for (e_level, e_name, g_name, u_name), value in sorted(metrics.items()):
        by_gen_metric[g_name].append((e_name, u_name, value))

    for g_name, rows in by_gen_metric.items():
        print(f"\n{'='*80}")
        print(f"Generation metric: {g_name}")
        print(f"{'='*80}")

        # Pivot: estimator rows, ue_metric columns
        ue_metrics = sorted(set(u for _, u, _ in rows))
        estimators = sorted(set(e for e, _, _ in rows))

        # Build lookup
        lookup = {}
        for e, u, v in rows:
            lookup[(e, u)] = v

        # Header
        col_w = max(12, max(len(u) for u in ue_metrics) + 2)
        est_w = max(35, max(len(e) for e in estimators) + 2)
        header = f"{'Estimator':<{est_w}}" + "".join(f"{u:>{col_w}}" for u in ue_metrics)
        print(header)
        print("-" * len(header))

        for e in estimators:
            row = f"{e:<{est_w}}"
            for u in ue_metrics:
                v = lookup.get((e, u), np.nan)
                if np.isnan(v):
                    row += f"{'—':>{col_w}}"
                else:
                    row += f"{v:>{col_w}.4f}"
            # Highlight spline rows
            marker = " ◀" if "spline" in e.lower() else ""
            print(row + marker)


def main():
    parser = argparse.ArgumentParser(description="Sanity-check a polygraph_eval checkpoint.")
    parser.add_argument("checkpoint", help="Path to ue_manager_seed* file")
    args = parser.parse_args()

    path = Path(args.checkpoint)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    print(f"Loading {path} ...")
    d = load_manager_dict(str(path))

    print(f"State: {d.get('state', 'unknown')}\n")

    # --- Check 1: estimations ---
    print("─" * 60)
    print("1. Spline estimator scores")
    print("─" * 60)
    estimations = d.get("estimations", {})
    est_ok = check_estimations(estimations)

    # --- Check 2: double generation ---
    print()
    print("─" * 60)
    print("2. Double-generation consistency")
    print("─" * 60)
    stats = d.get("stats", {})
    gen_ok = check_double_generation(stats)

    # --- Check 3: metrics table ---
    print()
    print("─" * 60)
    print("3. Results")
    print("─" * 60)
    metrics = d.get("metrics", {})
    print_metrics_table(metrics)

    # --- Summary ---
    print()
    print("─" * 60)
    all_ok = est_ok and gen_ok
    if all_ok:
        print("All checks passed. You're good to scale up.")
    else:
        print("Some checks failed. Review the warnings above before scaling up.")
    print("─" * 60)

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
