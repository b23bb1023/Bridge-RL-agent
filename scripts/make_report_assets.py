"""Generate the LaTeX table and number macros used by report/main.tex from the evaluation results.

Every number quoted in the report comes from here, so re-running the pipeline updates the paper.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS, OUT = ROOT / "results", ROOT / "report" / "generated"

ROWS = [
    ("DQN (ours)", "\\textbf{DQN (ours)}"),
    ("DQN (ours, 5-seed ensemble)", "DQN (ours), 5-seed ensemble"),
    ("Lookahead planner (teacher)", "Lookahead planner (teacher)"),
    ("DQN, offline pre-training only", "DQN, offline pre-training only"),
    ("DQN, rule demonstrations", "DQN, rule demonstrations"),
    ("DQN, no demonstrations", "DQN, no demonstrations"),
    ("DQN, shared network", "DQN, shared (undecomposed) net"),
    ("DQN, flat network", "DQN, flat MLP"),
    ("Maskable PPO", "Maskable PPO"),
    ("Vanilla DQN", "Vanilla DQN (no mask)"),
    ("DQN, last inspection only", "DQN, last inspection only"),
    ("DQN, no extra inspections", "DQN, no extra inspections"),
    ("DQN, true condition", "DQN, true condition$^\\dagger$"),
    ("Rule: condition-based", "Rule: condition-based"),
    ("Rule: entropy-triggered", "Rule: entropy-triggered"),
    ("Rule: risk-greedy", "Rule: risk-greedy"),
    ("Rule: oracle threshold", "Rule: oracle threshold$^\\dagger$"),
    ("DQN (random init, untrained)", "DQN, untrained"),
    ("Do nothing", "Do nothing"),
]


def money(v: float) -> str:
    v = v / 1e6
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.1f}" if abs(v) >= 20 else f"{v:,.2f}"


def money2(v: float) -> str:
    text = f"{v / 1e6:.2f}"
    return "\\textminus{}" + text[1:] if text.startswith("-") else text


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def main() -> None:
    import argparse
    global RESULTS, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out-dir", default="report/generated")
    args = parser.parse_args()
    RESULTS, OUT = ROOT / args.results_dir, ROOT / args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(RESULTS / "summary_nominal.csv")
    s = s[s["scenario"] == "nominal"].set_index("policy")

    lines = [
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "Policy & Cost \\$M [95\\% CI] & $\\Delta$ vs ours & Closed & Insp. \\\\",
        "\\midrule",
    ]
    for key, label in ROWS:
        if key not in s.index:
            continue
        r = s.loc[key]
        ci = f"{money(r['true_cost'])} [{money(r['true_cost_lo'])}, {money(r['true_cost_hi'])}]"
        if key == "DQN (ours)":
            diff = "--"
        elif pd.notna(r.get("diff_vs_main")):
            d = r["diff_vs_main"]
            diff = f"{'+' if d >= 0 else ''}{money2(d) if abs(d) < 10e6 else money(d)}"
        else:
            diff = ""
        lines.append(f"{label} & {ci} & {diff} & {r['closure_months']:.0f} & {r['inspections']:.1f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (OUT / "table1.tex").write_text("\n".join(lines) + "\n")

    # The same table in Markdown for the README.
    md = ["| Policy | 10-year cost, $M [95% CI] | Δ vs ours, $M [95% CI] | Bridge-months closed | Inspections |",
          "|---|---|---|---|---|"]
    for key, _ in ROWS:
        if key not in s.index:
            continue
        r = s.loc[key]
        diff = "—" if key == "DQN (ours)" or pd.isna(r.get("diff_vs_main")) else \
            f"{r['diff_vs_main'] / 1e6:+.2f} [{r['diff_lo'] / 1e6:+.2f}, {r['diff_hi'] / 1e6:+.2f}]"
        name = f"**{key}**" if key == "DQN (ours)" else key
        md.append(f"| {name} | {r['true_cost'] / 1e6:.2f} [{r['true_cost_lo'] / 1e6:.2f}, {r['true_cost_hi'] / 1e6:.2f}] "
                  f"| {diff} | {r['closure_months']:.1f} | {r['inspections']:.1f} |")
    (RESULTS / "table1.md").write_text("\n".join(md) + "\n")
    # Keep the README's results table in step with the results.
    readme = ROOT / "README.md"
    start, end = "<!-- results-table:start -->", "<!-- results-table:end -->"
    text = readme.read_text()
    if start in text and end in text:
        head, rest = text.split(start, 1)
        readme.write_text(head + start + "\n" + "\n".join(md) + "\n" + end + rest.split(end, 1)[1])

    def cost(key):
        return s.loc[key, "true_cost"] if key in s.index else float("nan")

    rules = [k for k in s.index if k.startswith("Rule:") and "oracle" not in k]
    best_rule = min(rules, key=cost)
    main_cost = cost("DQN (ours)")
    calib = yaml.safe_load((ROOT / "configs" / "calibration.yaml").read_text())
    m = ""
    m += macro("MainCost", money(main_cost))
    m += macro("MainLo", money(s.loc["DQN (ours)", "true_cost_lo"]))
    m += macro("MainHi", money(s.loc["DQN (ours)", "true_cost_hi"]))
    m += macro("BestRule", best_rule.replace("Rule: ", ""))
    m += macro("BestRuleCost", money(cost(best_rule)))
    m += macro("GainVsRule", f"{100 * (1 - main_cost / cost(best_rule)):.0f}")
    diff = s.loc[best_rule]
    # diff_vs_main is (policy - ours), so for a rule it is the saving our agent achieves.
    m += macro("GainVsRuleAbs", money2(diff["diff_vs_main"]))
    m += macro("GainVsRuleLo", money2(diff["diff_lo"]))
    m += macro("GainVsRuleHi", money2(diff["diff_hi"]))
    for key, name in [("DQN, flat network", "FlatCost"), ("Maskable PPO", "PPOCost"), ("DQN, true condition", "OracleCost"),
                      ("DQN, last inspection only", "LastObsCost"), ("DQN, no extra inspections", "NoInspCost"),
                      ("DQN, no demonstrations", "NoDemoCost"), ("DQN, shared network", "SharedCost"),
                      ("DQN, rule demonstrations", "RuleDemoCost"), ("DQN, offline pre-training only", "PretrainCost"), ("Lookahead planner (teacher)", "PlannerCost"),
                      ("DQN (ours, 5-seed ensemble)", "EnsembleCost"),
                      ("Do nothing", "NothingCost"), ("DQN (random init, untrained)", "RandomCost"),
                      ("Rule: oracle threshold", "OracleRuleCost"), ("Vanilla DQN", "VanillaCost")]:
        m += macro(name, money(cost(key)))
    m += macro("Funding", f"{calib['budget']['annual'] / 1e6:.2f}")
    m += macro("Sufficiency", f"{calib['calibration']['sufficiency_funding'] / 1e6:.2f}")
    m += macro("UnlimitedCost", money(calib["calibration"]["unlimited_cost"]))
    m += macro("Episodes", str(int(s.loc["DQN (ours)", "episodes"])))

    fit_path = ROOT / "data" / "processed" / "transition_fit.json"
    if fit_path.exists():
        fit = json.loads(fit_path.read_text())
        env_cfg = yaml.safe_load((ROOT / "configs" / "env.yaml").read_text())
        factor = float(env_cfg["deterioration"].get("unmaintained_factor", 1.0))
        years = fit["implied_mean_years_in_state"]
        m += macro("NPairs", f"{fit['diagnostics']['pairs_used']:,}")
        m += macro("RoutineSigma", f"{fit['routine_sigma']:.2f}")
        m += macro("AgeCoef", f"{fit['age_coef']:.2f}")
        m += macro("AdtCoef", f"{fit['adt_coef']:+.2f}")
        m += macro("ObservedYears", f"{min(years[1:]):.0f}--{max(years[1:]):.0f}")
        m += macro("ModelYears", f"{min(years[1:]) / factor:.0f}--{max(years[1:]) / factor:.0f}")
        m += macro("UnmaintainedFactor", f"{factor:g}")
    ood_path = RESULTS / "summary_ood.csv"
    if ood_path.exists():
        ood = pd.read_csv(ood_path)
        part = ood[(ood["policy"] == best_rule) & (ood["scenario"] != "nominal")]
        m += macro("OODTotal", str(len(part)))
        m += macro("OODWins", str(int((part["diff_lo"] > 0).sum())))
        m += macro("OODLosses", str(int((part["diff_hi"] < 0).sum())))
        ours = ood[ood["policy"] == "DQN (ours)"].set_index("scenario")["true_cost"]
        names = {"slower_decay": "Slow", "faster_decay": "Fast", "unmodelled_decay": "Unmod", "budget_cut": "Cut",
                 "noisy_routine": "Noisy", "unseen_networks": "Unseen"}
        for _, r in part.iterrows():
            name = names.get(r["scenario"])
            if name is None:
                continue
            # Saving of our agent over the best rule (positive = ours cheaper), and both costs.
            m += macro(f"{name}Saving", money2(r["diff_vs_main"]))
            m += macro(f"{name}Lo", money2(r["diff_lo"]))
            m += macro(f"{name}Hi", money2(r["diff_hi"]))
            m += macro(f"{name}Loss", money2(-r["diff_vs_main"]))
            m += macro(f"{name}Ours", money(ours.get(r["scenario"], float("nan"))))
            m += macro(f"{name}Rule", money(r["true_cost"]))
    ens = "DQN (ours, 5-seed ensemble)"
    if ens in s.index and pd.notna(s.loc[best_rule].get("diff_vs_main")):
        m += macro("EnsembleGain", f"{100 * (1 - cost(ens) / cost(best_rule)):.0f}")
    planner = "Lookahead planner (teacher)"
    if planner in s.index:
        m += macro("GainVsPlanner", money2(s.loc[planner, "diff_vs_main"]))
        m += macro("GainVsPlannerLo", money2(s.loc[planner, "diff_lo"]))
        m += macro("GainVsPlannerHi", money2(s.loc[planner, "diff_hi"]))
    # Paired differences (policy - ours) with CIs, for the comparisons quoted in the text.
    for key, name in [("DQN, true condition", "Oracle"), ("DQN, last inspection only", "LastObs"),
                      ("DQN, no extra inspections", "NoInsp"), ("DQN, offline pre-training only", "Pretrain"),
                      ("DQN, shared network", "Shared"), ("DQN (ours, 5-seed ensemble)", "Ensemble")]:
        if key in s.index:
            r = s.loc[key]
            m += macro(f"{name}Diff", money2(r["diff_vs_main"]))
            m += macro(f"{name}Lo", money2(r["diff_lo"]))
            m += macro(f"{name}Hi", money2(r["diff_hi"]))
            m += macro(f"{name}Gap", money2(-r["diff_vs_main"]))
            m += macro(f"{name}GapLo", money2(-r["diff_hi"]))
            m += macro(f"{name}GapHi", money2(-r["diff_lo"]))
    main_row = s.loc["DQN (ours)"]
    m += macro("MainInsp", f"{main_row['inspections']:.1f}")
    m += macro("MainClosed", f"{main_row['closure_months']:.1f}")
    m += macro("RuleClosed", f"{s.loc[best_rule, 'closure_months']:.1f}")
    for comp in ("agency", "closure", "risk", "salvage"):
        m += macro(f"Main{comp.capitalize()}", money(main_row[comp]))
        m += macro(f"Rule{comp.capitalize()}", money(s.loc[best_rule, comp]))
    demos = sorted((ROOT / "data" / "demos").glob("planner_budget1_belief.npz"))
    if demos:
        import numpy as np
        with np.load(demos[0]) as d:
            m += macro("DemoEpisodes", f"{len(d['action'])}")
            m += macro("PlannerDemoCost", money(float(d["true_cost"].mean())))
    timing_path = RESULTS / "decision_time.json"
    if timing_path.exists():
        timing = json.loads(timing_path.read_text())
        dqn_ms, plan_ms = 1e3 * timing.get("DQN (ours)", float("nan")), 1e3 * timing.get("Lookahead planner (teacher)", float("nan"))
        m += macro("DQNms", f"{dqn_ms:.2f}")
        m += macro("Plannerms", f"{plan_ms:,.0f}")
        m += macro("SpeedUp", f"{plan_ms / dqn_ms:,.0f}")
    (OUT / "numbers.tex").write_text(m)

    figs = OUT.parent / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    for f in (RESULTS / "figures").glob("*.pdf"):
        shutil.copy(f, figs / f.name)
    print(f"wrote {OUT / 'table1.tex'}, {OUT / 'numbers.tex'} and copied figures")


if __name__ == "__main__":
    main()
