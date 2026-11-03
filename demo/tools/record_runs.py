"""Record real Mesa runs of MountainRescueModel to JSON for the browser replay.

The configuration below matches SIMULATION_CONFIG in
KPI Analysis/Activity_4_KPI_Analysis.ipynb. Novel mode loads the pretrained
checkpoint and keeps the model's default behaviour, which is to continue training
during the run.

Usage:
    python record_runs.py --src <dir with model.py, agents.py, messaging.py, gnn_marl.py>
                          --checkpoint <best_gnn_marl_model.pth> --out <dir>
                          [--modes basic extended novel] [--runs 0 1 2 3 4]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

CONFIG = {
    "width": 15,
    "height": 15,
    "n_robots": 4,
    "n_drones": 3,
    "n_persons": 8,
    "spawn_interval": 30,
    "max_persons": 30,
    "steps_per_run": 500,
    "seed_base": 42,
}


def record(mode, run_id, checkpoint):
    import numpy as np
    import torch
    from agents import ExplorerDrone, FirstAidRobot, MobileCharger, Person
    from model import MountainRescueModel

    seed = CONFIG["seed_base"] + run_id * 100
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    kwargs = dict(
        width=CONFIG["width"],
        height=CONFIG["height"],
        n_robots=CONFIG["n_robots"],
        n_drones=CONFIG["n_drones"],
        n_persons=CONFIG["n_persons"],
        mode=mode,
        spawn_interval=CONFIG["spawn_interval"],
        max_persons=CONFIG["max_persons"],
        seed=seed,
        quiet=True,
    )
    if mode == "novel":
        # The fallback loader in model.py maps storages to self.device, so on a CPU-only machine the
        # checkpoint silently fails to load unless a device is passed.
        kwargs["pretrained_model_path"] = str(checkpoint)
        kwargs["device"] = torch.device("cpu")
    model = MountainRescueModel(**kwargs)
    if mode == "novel":
        system = model.gnn_marl_system
        assert system.total_steps > 0 and system.episode_count > 0, "checkpoint did not load"
        loaded = (int(system.episode_count), int(system.total_steps))

    kinds = ((FirstAidRobot, "robot"), (ExplorerDrone, "drone"), (MobileCharger, "charger"))
    movers = []
    for cls, kind in kinds:
        for agent in model.agents:
            if isinstance(agent, cls):
                movers.append((agent, kind))

    state_table = []

    def state_index(agent):
        value = agent.state.value
        if value not in state_table:
            state_table.append(value)
        return state_table.index(value)

    tracks = [{"kind": k, "id": a.unique_id, "x": [], "y": [], "s": [], "b": [],
               "cap": a.battery_capacity if hasattr(a, "battery_capacity") else None}
              for a, k in movers]
    persons = {}
    series = {"rescued": [], "messages": [], "persons": [], "modelSteps": []}
    if mode == "novel":
        series["epsilon"] = []

    def capture(t):
        for track, (agent, _) in zip(tracks, movers):
            x, y = agent.cell.coordinate
            track["x"].append(int(x))
            track["y"].append(int(y))
            track["s"].append(state_index(agent))
            track["b"].append(round(float(agent.battery), 1))
        for agent in model.agents:
            if isinstance(agent, Person):
                rec = persons.get(agent.unique_id)
                if rec is None:
                    x, y = agent.cell.coordinate
                    rec = {"id": agent.unique_id, "x": int(x), "y": int(y), "born": t,
                           "rescued": None, "urgency": round(float(agent.urgency), 2)}
                    persons[agent.unique_id] = rec
                if rec["rescued"] is None and agent.is_rescued:
                    rec["rescued"] = t
        series["rescued"].append(int(model.rescued_count))
        series["messages"].append(int(model.message_system.total_messages_sent)
                                  if model.message_system else 0)
        series["persons"].append(sum(1 for a in model.agents if isinstance(a, Person)))
        series["modelSteps"].append(int(model.steps))
        if mode == "novel":
            series["epsilon"].append(round(float(model.gnn_marl_system.robot_policy.epsilon), 4))

    started = time.time()
    capture(0)
    for t in range(1, CONFIG["steps_per_run"] + 1):
        model.step()
        capture(t)

    total = series["persons"][-1]
    final = {
        "rescued": int(model.rescued_count),
        "personsTotal": total,
        "rescueRate": model.rescued_count / max(1, total),
        "messages": series["messages"][-1],
        "spawnedDynamically": int(getattr(model, "persons_spawned_dynamically", 0)),
        "modelSteps": int(model.steps),
    }
    meta = {"mode": mode, "runId": run_id, "seed": seed, "config": CONFIG,
            "checkpointLoaded": mode == "novel", "wallSeconds": round(time.time() - started, 1)}
    if mode == "novel":
        meta["checkpointEpisodes"], meta["checkpointTrainingSteps"] = loaded
    terrain = [[int(v) for v in col] for col in model.terrain]
    return {"meta": meta, "final": final, "terrain": terrain, "base": list(model.base_position),
            "states": state_table, "tracks": tracks,
            "persons": sorted(persons.values(), key=lambda p: (p["born"], p["id"])),
            "series": series}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--modes", nargs="+", default=["basic", "extended", "novel"])
    p.add_argument("--runs", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="run ids; seed = 42 + 100 * id")
    args = p.parse_args()

    sys.path.insert(0, str(Path(args.src).resolve()))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        for run_id in args.runs:
            data = record(mode, run_id, Path(args.checkpoint).resolve())
            path = out / f"{mode}-{run_id}.json"
            path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
            f = data["final"]
            print(f"{mode} run {run_id} seed {data['meta']['seed']}: rescued {f['rescued']}/{f['personsTotal']} "
                  f"messages {f['messages']} in {data['meta']['wallSeconds']}s, {path.stat().st_size} bytes",
                  flush=True)


if __name__ == "__main__":
    main()
