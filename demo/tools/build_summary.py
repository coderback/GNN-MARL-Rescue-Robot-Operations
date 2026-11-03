"""Summarise recorded runs and the repo's training metrics into demo/data/summary.json.

    python demo/tools/build_summary.py
"""

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "demo" / "runs"
MODES = ["basic", "extended", "novel"]

runs, series = [], {m: [] for m in MODES}
for mode in MODES:
    for i in range(5):
        d = json.loads((RUNS / f"{mode}-{i}.json").read_text(encoding="utf8"))
        runs.append({"mode": mode, "runId": i, "seed": d["meta"]["seed"], **d["final"]})
        series[mode].append(d["series"]["rescued"])

steps = list(range(0, 501, 5))
mean_rescued = {m: [round(st.mean(r[s] for r in series[m]), 2) for s in steps] for m in MODES}
agg = {}
for m in MODES:
    rs = [r for r in runs if r["mode"] == m]
    agg[m] = {k: {"mean": st.mean(r[k] for r in rs), "std": st.stdev(r[k] for r in rs)} for k in ("rescued", "personsTotal", "rescueRate", "messages")}

tm = json.loads((ROOT / "results" / "training_metrics.json").read_text(encoding="utf8"))
rates = tm["rescue_rates"]
W = 25
rolling = [round(st.mean(rates[max(0, i - W + 1): i + 1]), 4) for i in range(len(rates))]
cm = json.loads((ROOT / "results_curriculum" / "training_metrics.json").read_text(encoding="utf8"))
cur = json.loads((ROOT / "results_curriculum" / "curriculum" / "curriculum_metrics.json").read_text(encoding="utf8"))
training = {
    "rates": [round(r, 4) for r in rates],
    "rolling": rolling,
    "episodes": len(rates),
    "first50": st.mean(rates[:50]),
    "last50": st.mean(rates[-50:]),
    "best": max(rates),
    "evaluationScores": tm["evaluation_scores"],
    "curriculum": {
        "episodes": cur["total_episodes"],
        "stagesCompleted": cur["stages_completed"],
        "stageHistory": cur["stage_history"],
        "regressions": cur["regression_count"],
        "currentStage": cur["current_stage_config"]["name"],
        "stageConfig": cur["current_stage_config"],
        "evaluationScores": cm["evaluation_scores"],
        "last30": st.mean(cm["rescue_rates"][-30:]),
    },
}
out = {"steps": steps, "meanRescued": mean_rescued, "runs": runs, "aggregate": agg, "training": training}
(ROOT / "demo" / "data").mkdir(exist_ok=True)
(ROOT / "demo" / "data" / "summary.json").write_text(json.dumps(out, separators=(",", ":")), encoding="utf8")
print(json.dumps({"aggregate": agg, "training": {k: v for k, v in training.items() if k not in ("rates", "rolling")}}, indent=1))
