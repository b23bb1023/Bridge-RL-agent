"""Report figures, drawn directly from the training logs and evaluation outputs.

  fig1_learning.pdf   validation cost during training (mean and 95% CI across seeds)
  fig2_information.pdf  (a) information ladder, (b) robustness under stress (or the funding sweep)
  fig3_timeline.pdf   one test episode of the main agent: expected condition, actions, account balance
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS, FIGS = ROOT / "results", ROOT / "results" / "figures"

SURFACE = "#fcfcfb"
INK, INK_2, INK_3 = "#0b0b0b", "#52514e", "#8a8984"
GRID = "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # fixed series order, colour-blind safe
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 1.4,
    "lines.solid_capstyle": "round", "legend.frameon": False,
})
MILLIONS = FuncFormatter(lambda v, _: "$0" if v == 0 else f"${v:,.0f}M" if v >= 1 else f"${v:.1f}M")


def _save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _seed_curves(run: str) -> pd.DataFrame | None:
    logs = [pd.read_csv(p) for p in sorted((RESULTS / "training_logs" / run).glob("seed*/log.csv"))]
    if not logs:
        return None
    grid = np.unique(np.concatenate([lg["step"].to_numpy() for lg in logs]))
    curves = np.stack([np.interp(grid, lg["step"], lg["val_true_cost"]) for lg in logs]) / 1e6
    mean = curves.mean(axis=0)
    half = 1.96 * curves.std(axis=0, ddof=1) / np.sqrt(len(logs)) if len(logs) > 1 else np.zeros_like(mean)
    return pd.DataFrame({"step": grid, "mean": mean, "lo": np.maximum(mean - half, 1e-3), "hi": mean + half, "n": len(logs)})


def _best_rule_validation_cost() -> tuple[str, float]:
    """Validation cost of the best tuned rule, on the same episodes as the training curves."""
    from scripts.make_demos import base_rule
    from bridgepulse.config import load_env_config
    from bridgepulse.env import BridgeNetworkEnv
    from bridgepulse.evaluation import VALIDATION_SEEDS, evaluate

    name, rule = base_rule(1.0, "belief", True)
    return name, float(evaluate(BridgeNetworkEnv(load_env_config()), rule, VALIDATION_SEEDS)["true_cost"].mean())


def fig_learning() -> None:
    fig, ax = plt.subplots(figsize=(3.45, 1.75))
    series = [("main", "DQN (ours), planner demos"), ("rule_demos", "DQN, rule demos"),
              ("no_demos", "DQN, no demos"), ("ppo", "Maskable PPO")]
    for color, (run, label) in zip(SERIES, series):
        c = _seed_curves(run)
        if c is None:
            continue
        ax.fill_between(c["step"] / 1e3, c["lo"], c["hi"], color=color, alpha=0.12, linewidth=0)
        ax.plot(c["step"] / 1e3, c["mean"], color=color, linewidth=1.4, label=f"{label} ({int(c['n'].iloc[0])})")
    name, y = _best_rule_validation_cost()
    ax.axhline(y / 1e6, color=INK_3, linewidth=0.9, linestyle=(0, (4, 3)))
    ax.annotate(f"best tuned rule ({name.replace('_', '-')}), ${y / 1e6:.1f}M", (1.0, y / 1e6), xycoords=("axes fraction", "data"),
                xytext=(-2, -3), textcoords="offset points", ha="right", va="top", color=INK_2, fontsize=6.3)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(MILLIONS)
    ax.set_xlabel("environment steps (thousands)")
    ax.set_ylabel("10-year cost, validation")
    ax.legend(loc="upper right", title="seeds in brackets", title_fontsize=6.3)
    _save(fig, "fig1_learning")


def fig_information(summary: pd.DataFrame | None, sweep: pd.DataFrame | None, ood: pd.DataFrame | None = None) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.1, 1.95), gridspec_kw={"width_ratios": [1.15, 1.0], "wspace": 0.45})
    if summary is not None:
        nominal = summary[summary["scenario"] == "nominal"].set_index("policy")
        ladder = [
            ("DQN, true condition", "Oracle: true condition"),
            ("DQN (ours)", "Bayesian belief + inspections"),
            ("DQN, last inspection only", "Last inspection result only"),
            ("DQN, no extra inspections", "Belief, no extra inspections"),
        ]
        rows = [(lab, nominal.loc[p]) for p, lab in ladder if p in nominal.index]
        y = np.arange(len(rows))[::-1]
        for yi, (lab, r) in zip(y, rows):
            mean, lo, hi = r["true_cost"] / 1e6, r["true_cost_lo"] / 1e6, r["true_cost_hi"] / 1e6
            ax_a.barh(yi, mean, height=0.55, color=SERIES[0], edgecolor=SURFACE, linewidth=0)
            ax_a.errorbar(mean, yi, xerr=[[mean - lo], [hi - mean]], color=INK, elinewidth=0.8, capsize=2)
            ax_a.annotate(f"${mean:.1f}M", (hi, yi), xytext=(3, 0), textcoords="offset points", va="center", color=INK_2, fontsize=6.5)
        ax_a.set_yticks(y)
        ax_a.set_yticklabels([lab for lab, _ in rows])
        ax_a.set_xlabel("10-year cost (test, 95% CI)")
        ax_a.xaxis.set_major_formatter(MILLIONS)
        ax_a.grid(axis="y", visible=False)
        ax_a.set_xlim(0, max(r["true_cost_hi"] for _, r in rows) / 1e6 * 1.3 if rows else 1)
        ax_a.set_title("(a) what the agent observes", loc="left", color=INK)
    if ood is not None and summary is not None:
        nominal = summary[summary["scenario"] == "nominal"].set_index("policy")
        rules = [p for p in nominal.index if p.startswith("Rule:") and "oracle" not in p]
        best = nominal.loc[rules, "true_cost"].idxmin()
        names = {"nominal": "Nominal", "slower_decay": "Slower decay (x0.5)", "faster_decay": "Faster decay (x5/3)",
                 "unmodelled_decay": "Unmodelled decay (x1.5)", "budget_cut": "Funding cut 25%",
                 "noisy_routine": "Noisy routine ratings", "unseen_networks": "Unseen bridge networks"}
        part = ood[ood["policy"] == best].set_index("scenario")
        rows = [(names[k], part.loc[k]) for k in names if k in part.index]
        y = np.arange(len(rows))[::-1]
        for yi, (lab, r) in zip(y, rows):
            # diff_vs_main is (rule - ours): positive means our agent is cheaper.
            mean, lo, hi = r["diff_vs_main"] / 1e6, r["diff_lo"] / 1e6, r["diff_hi"] / 1e6
            color = SERIES[0] if lo > 0 else (SERIES[1] if hi < 0 else INK_3)
            ax_b.errorbar(mean, yi, xerr=[[mean - lo], [hi - mean]], fmt="o", color=color, markersize=4,
                          markeredgecolor=SURFACE, markeredgewidth=0.8, elinewidth=1.0, capsize=2)
            right = mean >= 0
            ax_b.annotate(f"{mean:+.1f}", (hi if right else lo, yi), xytext=(4 if right else -4, 0), textcoords="offset points",
                          va="center", ha="left" if right else "right", color=INK_2, fontsize=6.3)
        ax_b.axvline(0, color=INK_2, linewidth=0.8)
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], marker="o", linestyle="", color=c, markersize=4, label=l)
                   for c, l in ((SERIES[0], "ours cheaper"), (INK_3, "no clear difference"), (SERIES[1], "rule cheaper"))]
        ax_b.legend(handles=handles, loc="lower right", fontsize=6.0, handletextpad=0.1, borderaxespad=0.2)
        lo_all, hi_all = part["diff_lo"].min() / 1e6, part["diff_hi"].max() / 1e6
        ax_b.set_xlim(lo_all - 0.25 * (hi_all - lo_all), hi_all + 0.15 * (hi_all - lo_all))
        ax_b.set_yticks(y)
        ax_b.set_yticklabels([lab for lab, _ in rows])
        ax_b.grid(axis="y", visible=False)
        ax_b.set_xlabel(f"saving vs {best.replace('Rule: ', '')} rule ($M, paired 95% CI)")
        ax_b.set_title("(b) robustness: fixed policies under stress", loc="left", color=INK)
    elif sweep is not None and len(sweep):
        rules = sweep[sweep["policy"].str.startswith("Rule:") & ~sweep["policy"].str.contains("oracle")]
        best_rule = rules.loc[rules.groupby("budget_scale")["true_cost"].idxmin()].assign(policy="Best tuned rule")
        dqn = sweep[sweep["policy"] == "DQN (ours)"]
        for color, (label, part) in zip(SERIES, [("DQN (ours)", dqn), ("Best tuned rule (re-tuned per level)", best_rule)]):
            part = part.sort_values("budget_scale")
            ax_b.fill_between(part["budget_scale"], part["lo"] / 1e6, part["hi"] / 1e6, color=color, alpha=0.12, linewidth=0)
            ax_b.plot(part["budget_scale"], part["true_cost"] / 1e6, color=color, marker="o", markersize=3.5,
                      markeredgecolor=SURFACE, markeredgewidth=0.8, label=label)
        ax_b.set_yscale("log")
        ax_b.yaxis.set_major_formatter(MILLIONS)
        ax_b.set_xlabel("yearly funding (x calibrated level)")
        ax_b.set_ylabel("10-year cost (test)")
        ax_b.legend(loc="upper right")
        ax_b.set_title("(b) funding sweep", loc="left", color=INK)
    _save(fig, "fig2_information")


def fig_timeline(tl: dict | None) -> None:
    if tl is None:
        return
    months = tl["months"]
    exp_c = np.array([m["expected"] for m in months]).T
    labels = [f"{a[:3]}-{i + 1}" for i, a in enumerate(tl["archetype"])]
    fig, (ax, ax_b) = plt.subplots(2, 1, figsize=(7.1, 2.55), sharex=True, gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.12})
    cmap = LinearSegmentedColormap.from_list("cond", BLUE_RAMP)
    im = ax.imshow(exp_c, aspect="auto", cmap=cmap, vmin=0, vmax=4, interpolation="nearest",
                   extent=(-0.5, len(months) - 0.5, len(labels) - 0.5, -0.5))
    markers = {1: ("o", "inspect"), 2: ("^", "minor repair"), 3: ("s", "major repair")}
    for kind, (mk, name) in markers.items():
        pts = [(t, m["bridge"]) for t, m in enumerate(months) if m["action_kind"] == kind]
        if pts:
            t, b = zip(*pts)
            ax.scatter(t, b, marker=mk, s=16, facecolor="white", edgecolor=INK, linewidth=0.7, label=name, zorder=3)
    closed = np.array([m["closed"] for m in months]).T
    for b, t in zip(*np.nonzero(closed)):
        ax.plot([t - 0.5, t + 0.5], [b, b], color=SERIES[1], linewidth=2.2, solid_capstyle="butt", zorder=2)
    ax.plot([], [], color=SERIES[1], linewidth=2.2, label="closed")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=6.3)
    ax.grid(False)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=4, fontsize=6.3, handletextpad=0.4, borderaxespad=0,
              columnspacing=1.2)
    cbar = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.025)
    cbar.set_label("expected condition\n(0 best, 4 critical)", fontsize=6.3, color=INK_2)
    cbar.ax.tick_params(labelsize=6.3)
    cbar.outline.set_visible(False)
    balance = np.array([m["balance"] for m in months]) / 1e6
    ax_b.plot(np.arange(len(months)), balance, color=SERIES[0], linewidth=1.3)
    ax_b.set_ylabel("account\n($M)", fontsize=6.5)
    ax_b.set_xlabel("month")
    pos_a, pos_b = ax.get_position(), ax_b.get_position()
    ax_b.set_position([pos_b.x0, pos_b.y0, pos_a.width, pos_b.height])
    _save(fig, "fig3_timeline")


def main() -> None:
    import argparse
    global RESULTS, FIGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    RESULTS = ROOT / args.results_dir
    FIGS = RESULTS / "figures"
    summary = pd.read_csv(RESULTS / "summary_nominal.csv") if (RESULTS / "summary_nominal.csv").exists() else None
    sweep = pd.read_csv(RESULTS / "summary_budget_sweep.csv") if (RESULTS / "summary_budget_sweep.csv").exists() else None
    ood = pd.read_csv(RESULTS / "summary_ood.csv") if (RESULTS / "summary_ood.csv").exists() else None
    tl = json.loads((RESULTS / "timeline.json").read_text()) if (RESULTS / "timeline.json").exists() else None
    fig_learning()
    fig_information(summary, sweep, ood)
    fig_timeline(tl)
    print(f"figures written to {FIGS}")


if __name__ == "__main__":
    main()
