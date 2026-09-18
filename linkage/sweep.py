"""Consistency sweep: PR-AUC vs repeat_rate (TECHNICAL_SPEC.md §9).

Regenerates a corpus at each repeat_rate in [0.2 .. 0.9], trains and
evaluates on each, and plots PR-AUC against repeat_rate with the prior
drawn on as a fixed reference line. This is the honesty-requirements
headline (CLAUDE.md): don't tune repeat_rate until the numbers look good —
show the whole curve, including where the method fails.

Runs on an 8,000-case corpus per point (not the 45k final corpus) purely so
eight end-to-end regenerate+train+evaluate passes finish in a few minutes;
n_cases and serial_case_fraction are held fixed across points, so the pool
composition (and hence the prior) barely moves — only repeat_rate
(consistency) changes.

    python -m linkage.sweep --out reports/consistency_sweep.png
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from linkage import config as config_mod
from linkage.evaluate import evaluate
from linkage.generate import corrupt, render, sample
from linkage.train import Context, train

REPEAT_RATES = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
N_CASES = 8000
OUT_PATH = Path(__file__).resolve().parent.parent / "reports" / "consistency_sweep.png"


def _generate(cfg: dict, repeat_rate: float, seed: int, out: Path) -> None:
    cfg = {**cfg, "corpus": {**cfg["corpus"], "n_cases": N_CASES,
                              "repeat_rate": {**cfg["corpus"]["repeat_rate"], "value": repeat_rate}}}
    readiness = config_mod.check(cfg)
    if readiness.errors:
        raise RuntimeError(f"config invalid at repeat_rate={repeat_rate}: {readiness.errors}")

    truth, _offenders = sample.generate(cfg, seed)
    model = sample.build_model(cfg)
    cases = truth.merge(corrupt.record(truth, cfg, seed, model), on="case_uid")
    feeds, _positions = render.feeds(cases, cfg, cfg["vocab"], seed)

    (out / "feeds").mkdir(parents=True, exist_ok=True)
    cases.to_parquet(out / "truth.parquet", index=False)
    for code, frame in feeds.items():
        frame.to_csv(out / "feeds" / f"{code}.csv", index=False, encoding="utf-8")


def run(cfg: dict, seed: int = 7) -> list[dict]:
    results = []
    with tempfile.TemporaryDirectory(prefix="linkage_sweep_") as tmp:
        tmp_path = Path(tmp)
        for rr in REPEAT_RATES:
            point_dir = tmp_path / f"rr_{rr}"
            print(f"repeat_rate={rr}: generating {N_CASES} cases...")
            _generate(cfg, rr, seed, point_dir)

            point_cfg = {**cfg, "corpus": {**cfg["corpus"], "n_cases": N_CASES,
                                           "repeat_rate": {**cfg["corpus"]["repeat_rate"], "value": rr}}}
            ctx = Context(point_dir, point_cfg, seed=0, embed_dim=256)
            weights = train(ctx)
            metrics = evaluate(ctx, weights)
            prior_bits = ctx.priors["same_type"].get("BURGLARY_RESIDENTIAL")
            results.append({
                "repeat_rate": rr,
                "pr_auc_same_type": metrics["pr_auc"]["same_type"],
                "pr_auc_cross_type": metrics["pr_auc"]["cross_type"],
                "prior_bits": prior_bits,
            })
            print(f"  PR-AUC same_type={metrics['pr_auc']['same_type']} "
                  f"cross_type={metrics['pr_auc']['cross_type']} prior={prior_bits}")
    return results


def plot(results: list[dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rr = [r["repeat_rate"] for r in results]
    same = [r["pr_auc_same_type"] or 0.0 for r in results]
    cross = [r["pr_auc_cross_type"] or 0.0 for r in results]
    prior = next((r["prior_bits"] for r in results if r["prior_bits"] is not None), None)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(rr, same, marker="o", color="#2b6cb0", label="PR-AUC (same_type)")
    ax1.plot(rr, cross, marker="s", color="#c05621", label="PR-AUC (cross_type)")
    ax1.set_xlabel("repeat_rate (target m at reference_frequency)")
    ax1.set_ylabel("PR-AUC")
    ax1.set_ylim(0, 1)
    ax1.set_title("Consistency sweep: how behaviourally consistent an offender\n"
                  "must be before MO evidence alone finds them")

    if prior is not None:
        ax2 = ax1.twinx()
        ax2.axhline(prior, color="gray", linestyle="--", linewidth=1)
        ax2.set_ylabel("prior (bits, same_type burglary_residential)")
        ax2.annotate(f"prior = {prior:.2f} bits", xy=(rr[-1], prior), xytext=(-5, 6),
                    textcoords="offset points", ha="right", color="gray", fontsize=9)

    ax1.legend(loc="upper left")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m linkage.sweep", description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--config-dir", type=Path, default=config_mod.CONFIG_DIR)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    cfg = config_mod.load(args.config_dir)
    results = run(cfg, seed=args.seed)
    plot(results, args.out)
    print(f"\nwrote {args.out}")
    for r in results:
        print(f"  repeat_rate={r['repeat_rate']}  same_type={r['pr_auc_same_type']}  "
              f"cross_type={r['pr_auc_cross_type']}  prior={r['prior_bits']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
