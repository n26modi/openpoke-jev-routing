"""Generate presentation charts from a benchmark run.

Visual style borrows directly from General Magic's blog chart grammar:
Inter typography, warm off-white background, orange-red accent on the
winner, muted neutrals on the comparisons.

Usage:
    python -m server.eval.charts            # uses cached results
    python -m server.eval.charts --rerun    # re-executes the benchmark first

Writes PNGs to `server/eval/out/`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt

from .run import CONFIGS, _run_case, _summarize
from .benchmark import BENCHMARK


_OUT_DIR = Path(__file__).parent / "out"
_FONT_DIR = Path(__file__).parent / "fonts" / "Inter" / "extras" / "ttf"


# ---- GM palette ----
# Pulled from the HTML of https://generalmagic.inc/blog/insurance-specific-ai
ACCENT      = "#FF4B33"   # winner (vivid orange-red, the brand accent)
ACCENT_SOFT = "#FF8A70"   # accent at reduced saturation
DARK        = "#111113"   # near-black text
NEUTRAL_MID = "#75757B"   # body text / muted labels
NEUTRAL_LT  = "#A9B2BB"   # middle bar (baseline — "OpenPoke today")
NEUTRAL_XLT = "#DDE2E7"   # lightest bar (bottom tier)
GRID        = "#ECECEF"   # gridlines
BG          = "#FFF6F4"   # warm off-white page background


_PALETTE = {
    "baseline":       NEUTRAL_LT,
    "hot_cache_only": NEUTRAL_XLT,
    "full_hybrid":    ACCENT,
}
_LABELS = {
    "baseline":       "Baseline (OpenPoke today)",
    "hot_cache_only": "Hot cache only",
    "full_hybrid":    "Full hybrid (Jev + semantic)",
}


# ---- font setup ----

def _register_inter() -> str:
    """Register bundled Inter TTFs and return the family name to use."""
    if not _FONT_DIR.exists():
        return "DejaVu Sans"
    for ttf in _FONT_DIR.glob("Inter-*.ttf"):
        fm.fontManager.addfont(str(ttf))
    return "Inter"


_FAMILY = _register_inter()
mpl.rcParams.update({
    "font.family": _FAMILY,
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "axes.edgecolor": NEUTRAL_MID,
    "axes.linewidth": 0.6,
    "axes.titleweight": "bold",
    "axes.titlecolor": DARK,
    "axes.labelcolor": NEUTRAL_MID,
    "xtick.color": NEUTRAL_MID,
    "ytick.color": DARK,
    "xtick.labelsize": 10,
    "ytick.labelsize": 11,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "savefig.edgecolor": BG,
})


# ---- benchmark execution + cache ----

def _run_benchmark() -> list[dict]:
    from ..services.retrieval import is_jev_available
    jev_on = is_jev_available()
    results = []
    for cfg in CONFIGS:
        if cfg.use_jev_rerank and not jev_on:
            continue
        for case in BENCHMARK:
            results.append(_run_case(cfg, case))
    summaries = _summarize(results)
    return [
        {
            "config": s.name,
            "accuracy": s.accuracy,
            "avg_visible": s.avg_visible,
            "avg_latency_ms": s.avg_latency_ms,
        }
        for s in summaries
    ]


def _load_cached(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _save_cache(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2))


# ---- shared chart styling ----

def _style_bar_axes(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(NEUTRAL_MID)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="y", length=0, pad=10)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.grid(axis="x", color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def _style_scatter_axes(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(NEUTRAL_MID)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(length=0, pad=6)
    ax.grid(color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


# ---- chart 1: accuracy ----

def chart_accuracy(rows: list[dict], out_path: Path) -> None:
    baseline = next(r for r in rows if r["config"] == "baseline")
    base_acc = baseline["accuracy"] or 1e-9

    ordered = sorted(rows, key=lambda r: r["accuracy"])
    labels = [_LABELS[r["config"]] for r in ordered]
    values = [r["accuracy"] * 100 for r in ordered]
    colors = [_PALETTE[r["config"]] for r in ordered]

    # Extend x-axis past 100% to leave a dedicated column for the labels.
    xmax = 150
    fig, ax = plt.subplots(figsize=(10.5, 4.0), dpi=200)
    bars = ax.barh(labels, values, color=colors, zorder=2, height=0.58)
    _style_bar_axes(ax)
    ax.set_xlim(0, xmax)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("accuracy on 13-case routing benchmark",
                  fontsize=10, labelpad=10)
    ax.set_title("Accuracy", loc="left", pad=16, fontsize=18, color=DARK)

    # Fixed label column: bold percentage at x=102, tag at x=115. All rows aligned.
    for bar, row in zip(bars, ordered):
        w = bar.get_width()
        cfg = row["config"]
        pct = f"{w:.1f}%"
        if cfg == "baseline":
            tag = "OpenPoke today"
        else:
            mult = (w / 100) / base_acc
            tag = (f"{mult:.2f}× more accurate" if mult > 1
                   else f"{mult:.2f}× — worse")

        y = bar.get_y() + bar.get_height() / 2
        ax.text(102, y, pct, va="center", ha="left",
                color=DARK, fontsize=13, fontweight="bold")
        ax.text(118, y, tag, va="center", ha="left",
                color=NEUTRAL_MID, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---- chart 2: visible roster size ----

def chart_visible(rows: list[dict], out_path: Path) -> None:
    baseline = next(r for r in rows if r["config"] == "baseline")
    base_vis = baseline["avg_visible"] or 1e-9

    ordered = sorted(rows, key=lambda r: -r["avg_visible"])  # worst (longest) at top
    labels = [_LABELS[r["config"]] for r in ordered]
    values = [r["avg_visible"] for r in ordered]
    colors = [_PALETTE[r["config"]] for r in ordered]

    # x-axis ends at max value; label column sits to the right in a padded area.
    data_max = max(values)
    xmax = data_max * 1.8
    fig, ax = plt.subplots(figsize=(10.5, 4.0), dpi=200)
    bars = ax.barh(labels, values, color=colors, zorder=2, height=0.58)
    _style_bar_axes(ax)
    ax.set_xlim(0, xmax)
    # Only tick to data_max; dedicated label column lives past it.
    ax.set_xticks([0, 5, 10, 15, 20, 25])
    ax.set_xlabel("avg. agents visible to Interaction Agent (lower is better)",
                  fontsize=10, labelpad=10)
    ax.set_title("Roster noise per turn", loc="left", pad=16, fontsize=18, color=DARK)

    num_x = data_max * 1.1
    tag_x = data_max * 1.25
    for bar, row in zip(bars, ordered):
        w = bar.get_width()
        cfg = row["config"]
        num = f"{w:.1f}"
        if cfg == "baseline":
            tag = "OpenPoke today"
        else:
            reduction = base_vis / w if w > 0 else float("inf")
            tag = f"{reduction:.1f}× less noise"

        y = bar.get_y() + bar.get_height() / 2
        ax.text(num_x, y, num, va="center", ha="left",
                color=DARK, fontsize=13, fontweight="bold")
        ax.text(tag_x, y, tag, va="center", ha="left",
                color=NEUTRAL_MID, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---- chart 3: scatter ----

def chart_scatter(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 6.4), dpi=200)
    _style_scatter_axes(ax)

    for r in rows:
        is_winner = r["config"] == "full_hybrid"
        ax.scatter(
            r["accuracy"] * 100, r["avg_visible"],
            s=420 if is_winner else 320,
            color=_PALETTE[r["config"]],
            edgecolor=BG, linewidth=3.0, zorder=3,
        )

    # Label placement: push labels below each point so they can't collide with
    # the title or spill over the axis frame.
    label_placement = {
        "baseline":       dict(dx=14,  dy=-6,  ha="left",  va="top"),
        "hot_cache_only": dict(dx=14,  dy=-6,  ha="left",  va="top"),
        "full_hybrid":    dict(dx=-14, dy=-6,  ha="right", va="top"),
    }
    for r in rows:
        p = label_placement[r["config"]]
        ax.annotate(
            _LABELS[r["config"]],
            xy=(r["accuracy"] * 100, r["avg_visible"]),
            xytext=(p["dx"], p["dy"]), textcoords="offset points",
            ha=p["ha"], va=p["va"],
            color=DARK, fontsize=11, fontweight="bold",
        )

    ax.invert_yaxis()
    ax.set_xlim(45, 100)
    ax.set_ylim(28, -2)   # explicit so title has clear headroom
    ax.set_xlabel("accuracy (higher is better)", fontsize=10, labelpad=10)
    ax.set_ylabel("agents shown per turn (lower is better)",
                  fontsize=10, labelpad=12)
    ax.set_title("Accuracy vs. roster noise",
                 loc="left", pad=16, fontsize=18, color=DARK)

    # Soft "ideal" callout in the top-right corner
    ax.text(98, 0.5, "ideal", color=NEUTRAL_MID, fontsize=10,
            ha="right", va="top", style="italic")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---- main ----

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rerun", action="store_true",
                        help="Re-execute the benchmark instead of using cached results.")
    args = parser.parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = _OUT_DIR / "results.json"

    rows = None if args.rerun else _load_cached(cache)
    if rows is None:
        print("Running benchmark...")
        rows = _run_benchmark()
        _save_cache(rows, cache)
        print(f"cached results at {cache}")
    else:
        print(f"using cached results from {cache}")

    chart_accuracy(rows, _OUT_DIR / "chart_accuracy.png")
    chart_visible(rows, _OUT_DIR / "chart_visible.png")
    chart_scatter(rows, _OUT_DIR / "chart_scatter.png")
    print("done.")


if __name__ == "__main__":
    main()
