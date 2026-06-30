#!/usr/bin/env python3
"""
analyze_newton_log_v2.py — Phase 5.11.J analyzer

Reads the per-Newton-iter CSV emitted by SaddleNewtonDiagnosticLogger
and produces diagnostic summaries + plots showing:

  - Per-step convergence trajectories of the Newton residual
  - Physical block decomposition: K-block vs constraint vs
    per-sub-block constraint
  - Active scaling factor evolution across steps
  - Per-step summary table (initial / final residuals, iter count,
    convergence verdict, factor changes)
  - Anomaly detection: residual stalls, factor jumps, sub-block
    imbalance

Usage:

    python3 analyze_newton_log_v2.py newton_iters.csv               # summary table
    python3 analyze_newton_log_v2.py newton_iters.csv --plot        # + PNG plots
    python3 analyze_newton_log_v2.py newton_iters.csv --plot --out_dir plots/
    python3 analyze_newton_log_v2.py newton_iters.csv --steps 0,1,5 # only some steps
    python3 analyze_newton_log_v2.py newton_iters.csv --watch       # tail mode

Header format (column count varies by partition):

    step, iter,
    norm, norm0, norm_max, converged_now, scaler_enabled,
    res_K, res_lam,
    res_lam_<label_0>, ..., res_lam_<label_{N-1}>,
    d_u,
    d_lam_<label_0>, ..., d_lam_<label_{N-1}>

The label list is detected from the header on read.
"""

import argparse
import csv
import math
import os
import sys
import time
from collections import defaultdict


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------

def read_csv(path):
    """Read the CSV, returning a dict with keys 'header', 'rows',
    'sub_labels'. Each row is a dict mapping column name -> value
    (numeric where appropriate)."""
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        rows = list(reader)

    if not header:
        raise ValueError(f"empty or unreadable CSV: {path}")

    # Detect sub-block labels from the 'res_lam_*' column prefix.
    sub_labels = []
    for name in header:
        if name.startswith("res_lam_"):
            sub_labels.append(name[len("res_lam_"):])

    # Convert numeric fields.
    int_fields = {"step", "iter", "converged_now", "scaler_enabled"}
    float_fields = {"norm", "norm0", "norm_max", "res_K", "res_lam", "d_u"}
    for label in sub_labels:
        float_fields.add(f"res_lam_{label}")
        float_fields.add(f"d_lam_{label}")

    parsed_rows = []
    for raw in rows:
        out = {}
        for key, val in raw.items():
            if key in int_fields:
                try:
                    out[key] = int(val)
                except (TypeError, ValueError):
                    out[key] = -1
            elif key in float_fields:
                try:
                    out[key] = float(val)
                except (TypeError, ValueError):
                    out[key] = float("nan")
            else:
                out[key] = val
        parsed_rows.append(out)

    return {
        "header": header,
        "rows": parsed_rows,
        "sub_labels": sub_labels,
    }


def group_by_step(rows):
    """Return {step_index: [row, row, ...]} sorted by iter within each step."""
    by_step = defaultdict(list)
    for r in rows:
        by_step[r["step"]].append(r)
    for step in by_step:
        by_step[step].sort(key=lambda r: r["iter"])
    return dict(by_step)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def format_sci(x, digits=2):
    if x is None or (isinstance(x, float) and (math.isnan(x) or x < 0)):
        return f"{'--':>{digits+6}}"
    return f"{x:.{digits}e}"


def print_summary_table(by_step, sub_labels):
    """Per-step summary printed to stdout. Columns:
        step | iters | norm0 | norm_final | conv | res_K_init | res_lam_init | d_u | d_lam_*"""
    print()
    print("=" * 110)
    print("PER-STEP SUMMARY")
    print("=" * 110)

    # Fixed column widths for readability.
    header_cols = [
        ("step", 4),
        ("iters", 5),
        ("norm0", 10),
        ("norm_fin", 10),
        ("conv", 4),
        ("res_K_0", 10),
        ("res_lam_0", 10),
        ("d_u", 9),
    ]
    for lbl in sub_labels:
        header_cols.append((f"d_{lbl}", 9))

    fmt = "  ".join(f"{{:>{w}}}" for _, w in header_cols)
    print(fmt.format(*[h for h, _ in header_cols]))
    print("-" * 110)

    for step in sorted(by_step.keys()):
        iters = by_step[step]
        if not iters:
            continue
        first = iters[0]
        last = iters[-1]
        n_iter = len(iters)
        converged = last["converged_now"] == 1
        norm0 = first["norm"]
        norm_fin = last["norm"]
        res_K0 = first.get("res_K", float("nan"))
        res_lam0 = first.get("res_lam", float("nan"))
        d_u = first.get("d_u", float("nan"))
        d_lams = [first.get(f"d_lam_{lbl}", float("nan")) for lbl in sub_labels]

        row_vals = [
            str(step),
            str(n_iter),
            format_sci(norm0),
            format_sci(norm_fin),
            "yes" if converged else "NO",
            format_sci(res_K0),
            format_sci(res_lam0),
            format_sci(d_u),
        ]
        for d_lam in d_lams:
            row_vals.append(format_sci(d_lam))
        print(fmt.format(*row_vals))

    print("=" * 110)


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def detect_anomalies(by_step, sub_labels, factor_jump_threshold=10.0,
                      stall_ratio=0.99, stall_min_iters=3):
    """Print flagged patterns:
      - Steps where Newton didn't converge.
      - Steps where the residual stalled (last `stall_min_iters` ratios > stall_ratio).
      - Steps where d_u or any d_lam_* jumped by > factor_jump_threshold
        relative to the previous step.
      - Steps where the per-sub-block residual is dominated by one
        sub-block (one sub-block >> others), suggesting that sub-block
        is the bottleneck."""

    anomalies = []

    sorted_steps = sorted(by_step.keys())

    # Stalls and non-convergence per step.
    for step in sorted_steps:
        iters = by_step[step]
        if not iters:
            continue
        last = iters[-1]
        if last["converged_now"] != 1:
            anomalies.append(
                f"  step {step}: did NOT converge "
                f"(last norm = {format_sci(last['norm'])} vs threshold "
                f"{format_sci(last['norm_max'])})"
            )

        if len(iters) >= stall_min_iters + 1:
            # Compute consecutive ratios of norm[i] / norm[i-1] over the
            # tail. If they're all close to 1 the residual is stalled.
            tail = iters[-(stall_min_iters + 1):]
            ratios = []
            for i in range(1, len(tail)):
                a = tail[i]["norm"]
                b = tail[i - 1]["norm"]
                if b > 0 and not math.isnan(a) and not math.isnan(b):
                    ratios.append(a / b)
            if ratios and all(r > stall_ratio for r in ratios):
                anomalies.append(
                    f"  step {step}: residual STALLED — last "
                    f"{len(ratios)} ratios "
                    f"[{', '.join(f'{r:.3f}' for r in ratios)}] "
                    f"all > {stall_ratio}"
                )

    # Factor jumps between consecutive steps.
    factor_keys = ["d_u"] + [f"d_lam_{lbl}" for lbl in sub_labels]
    prev_factors = None
    prev_step = None
    for step in sorted_steps:
        iters = by_step[step]
        if not iters:
            continue
        first = iters[0]
        factors = {k: first.get(k, float("nan")) for k in factor_keys}
        if prev_factors is not None:
            for k in factor_keys:
                a = factors[k]
                b = prev_factors[k]
                if (a > 0 and b > 0 and not math.isnan(a)
                        and not math.isnan(b)):
                    ratio = max(a / b, b / a)
                    if ratio > factor_jump_threshold:
                        anomalies.append(
                            f"  step {prev_step}->{step}: {k} JUMPED "
                            f"by factor {ratio:.2g} "
                            f"({format_sci(b)} -> {format_sci(a)})"
                        )
        prev_factors = factors
        prev_step = step

    # Sub-block dominance — when one sub-block's residual is much
    # larger than the others at iter 0 of each step. This is just
    # informational; sub-block-aware scaling would target it.
    if sub_labels:
        for step in sorted_steps:
            iters = by_step[step]
            if not iters:
                continue
            first = iters[0]
            sub_norms = [first.get(f"res_lam_{lbl}", 0.0)
                          for lbl in sub_labels]
            valid = [(lbl, n) for lbl, n in zip(sub_labels, sub_norms)
                     if n > 0 and not math.isnan(n)]
            if len(valid) < 2:
                continue
            n_max = max(n for _, n in valid)
            n_min = min(n for _, n in valid)
            if n_max / max(n_min, 1e-30) > 100.0:
                dom_lbl = next(lbl for lbl, n in valid if n == n_max)
                anomalies.append(
                    f"  step {step}: sub-block '{dom_lbl}' dominates "
                    f"(max/min ratio = {n_max/n_min:.2g}) — sub-block "
                    f"scaling may help"
                )

    print()
    print("=" * 110)
    print("ANOMALIES")
    print("=" * 110)
    if not anomalies:
        print("  (none detected)")
    else:
        for line in anomalies:
            print(line)
    print("=" * 110)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plots(by_step, sub_labels, out_dir, only_steps=None):
    """Produce four PNGs in out_dir:
      - newton_residual_vs_iter.png    : ||r|| per iter, one line per step
      - per_block_residual_vs_iter.png : res_K, res_lam, per-sub-block on log y
      - scaling_factors_vs_step.png    : d_u + d_lam_* across steps
      - per_step_iter_count.png        : iters required per step (bar)"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[analyze] matplotlib not available; skipping plots", file=sys.stderr)
        return

    os.makedirs(out_dir, exist_ok=True)

    sorted_steps = sorted(by_step.keys())
    if only_steps is not None:
        sorted_steps = [s for s in sorted_steps if s in only_steps]
    if not sorted_steps:
        print("[analyze] no steps to plot", file=sys.stderr)
        return

    # ---- Plot 1: Newton residual vs iter, faceted by step ----
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.cm.viridis
    n_steps = len(sorted_steps)
    for i, step in enumerate(sorted_steps):
        iters = by_step[step]
        xs = [r["iter"] for r in iters]
        ys = [r["norm"] for r in iters]
        color = cmap(i / max(1, n_steps - 1))
        ax.semilogy(xs, ys, marker="o", color=color, label=f"step {step}",
                     linewidth=1.0, markersize=3)
    ax.set_xlabel("Newton iter")
    ax.set_ylabel("||r||  (scaled coords if scaling active)")
    ax.set_title("Newton residual evolution per step")
    if n_steps <= 12:
        ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, "newton_residual_vs_iter.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")

    # ---- Plot 2: per-block residual vs iter, faceted by step ----
    # One subplot per step (up to a max), each with res_K, res_lam,
    # and per-sub-block lambda on log y.
    n_plot = min(len(sorted_steps), 9)   # cap at 9 (3x3 grid)
    steps_to_plot = sorted_steps[:n_plot]
    n_cols = min(n_plot, 3)
    n_rows = (n_plot + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(4 * n_cols, 3 * n_rows),
                              sharey=True)
    if n_plot == 1:
        axes = [axes]
    else:
        axes = list(axes.flat) if hasattr(axes, "flat") else list(axes)
    for ax, step in zip(axes, steps_to_plot):
        iters = by_step[step]
        xs = [r["iter"] for r in iters]
        ax.semilogy(xs, [r.get("res_K", float("nan")) for r in iters],
                     marker="o", label="K-block", linewidth=1.5, markersize=3)
        ax.semilogy(xs, [r.get("res_lam", float("nan")) for r in iters],
                     marker="s", label="lambda (all)", linewidth=1.5,
                     markersize=3)
        for lbl in sub_labels:
            ax.semilogy(xs, [r.get(f"res_lam_{lbl}", float("nan"))
                              for r in iters],
                         marker=".", label=f"lam_{lbl}", linewidth=0.8,
                         linestyle="--", markersize=2)
        ax.set_title(f"step {step}", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        ax.set_xlabel("iter", fontsize=8)
    for ax in axes[n_plot:]:
        ax.axis("off")
    axes[0].set_ylabel("||r_*||  (physical)", fontsize=9)
    axes[0].legend(loc="best", fontsize=7)
    fig.suptitle("Per-block physical residual evolution")
    fig.tight_layout()
    out = os.path.join(out_dir, "per_block_residual_vs_iter.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")

    # ---- Plot 3: scaling factors across steps ----
    fig, ax = plt.subplots(figsize=(8, 5))
    step_xs = sorted_steps
    d_u_ys = [by_step[s][0].get("d_u", float("nan")) for s in step_xs]
    ax.semilogy(step_xs, d_u_ys, marker="o", label="d_u", linewidth=1.5)
    for lbl in sub_labels:
        ys = [by_step[s][0].get(f"d_lam_{lbl}", float("nan"))
              for s in step_xs]
        ax.semilogy(step_xs, ys, marker="s", label=f"d_lam_{lbl}",
                     linewidth=1.0, markersize=3)
    ax.set_xlabel("step")
    ax.set_ylabel("active scaling factor")
    ax.set_title("Saddle scaling factor evolution across steps")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, "scaling_factors_vs_step.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")

    # ---- Plot 4: iter count per step (bar) ----
    fig, ax = plt.subplots(figsize=(8, 4))
    iter_counts = [len(by_step[s]) for s in step_xs]
    converged = [by_step[s][-1]["converged_now"] == 1 for s in step_xs]
    bar_colors = ["tab:blue" if c else "tab:red" for c in converged]
    ax.bar(step_xs, iter_counts, color=bar_colors)
    ax.set_xlabel("step")
    ax.set_ylabel("Newton iters")
    ax.set_title("Iter count per step (red = did not converge)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(out_dir, "per_step_iter_count.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="path to newton_iters.csv")
    ap.add_argument("--plot", action="store_true",
                     help="produce PNG plots in --out_dir")
    ap.add_argument("--out_dir", default="newton_diag_plots",
                     help="output directory for plots (default: newton_diag_plots)")
    ap.add_argument("--steps", default=None,
                     help="comma-separated list of step indices to focus on, "
                          "e.g. '0,1,5'. Default: all.")
    ap.add_argument("--no_anomalies", action="store_true",
                     help="skip the anomaly-detection section")
    ap.add_argument("--watch", action="store_true",
                     help="tail mode: re-read every 5s and re-print summary")
    args = ap.parse_args(argv)

    if args.steps:
        only_steps = set(int(s) for s in args.steps.split(","))
    else:
        only_steps = None

    def run_once():
        try:
            data = read_csv(args.csv)
        except Exception as e:
            print(f"[analyze] ERROR reading {args.csv}: {e}", file=sys.stderr)
            return 1

        rows = data["rows"]
        if only_steps is not None:
            rows = [r for r in rows if r["step"] in only_steps]
        if not rows:
            print(f"[analyze] no rows in {args.csv}", file=sys.stderr)
            return 1

        sub_labels = data["sub_labels"]
        print(f"[analyze] read {len(rows)} rows from {args.csv}")
        print(f"[analyze] detected {len(sub_labels)} sub-block label(s): "
               f"{sub_labels if sub_labels else '(none)'}")

        by_step = group_by_step(rows)
        print_summary_table(by_step, sub_labels)

        if not args.no_anomalies:
            detect_anomalies(by_step, sub_labels)

        if args.plot:
            print(f"\n[analyze] plotting to {args.out_dir}/")
            make_plots(by_step, sub_labels, args.out_dir, only_steps=only_steps)

        return 0

    if not args.watch:
        return run_once()

    print("[analyze] watch mode — Ctrl-C to stop")
    while True:
        rc = run_once()
        if rc != 0:
            return rc
        time.sleep(5.0)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
