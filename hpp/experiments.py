"""Reproducible runs and paired ensembles; all reported data are computed here."""

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .model import collide, initialize, measure, momentum, stream, wall_impulse


def run(width=128, height=96, steps=800, boundary="periodic", p_left=0.45,
        p_right=0.05, seed=2026, block=8, snapshot_times=()):
    state = initialize(width, height, p_left, p_right, seed)
    mass0, p0 = int(state.sum()), momentum(state)
    impulse = np.zeros(2, dtype=np.int64)
    times = set(snapshot_times)
    rows, snapshots = [], {}
    for t in range(steps + 1):
        row = measure(state, t, block, mass0, p0, impulse)
        if row.mass_error or row.momentum_error:
            raise RuntimeError(f"Conservation law failed at step {t}: {row}")
        rows.append(asdict(row))
        if t in times:
            snapshots[t] = state.copy()
        if t < steps:
            post_collision = collide(state)
            impulse += wall_impulse(post_collision, boundary)
            state = stream(post_collision, boundary)
    return rows, snapshots, state


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def stable_threshold(values, threshold=0.1, window=25):
    """First start of a window whose every value is below threshold; no claim of permanence."""
    passed = np.asarray(values) <= threshold
    for t in range(len(passed) - window + 1):
        if passed[t:t + window].all():
            return t
    return None


def study(output: Path, width=128, height=96, steps=800, seeds=12, block=8):
    output.mkdir(parents=True, exist_ok=True)
    if seeds < 2:
        raise ValueError("At least two initial states are required for ensemble SD.")
    config = dict(width=width, height=height, steps=steps, seeds=seeds, block=block,
                  p_left=0.45, p_right=0.05, seed_start=2026, threshold=0.1, window=25)
    summary = {"config": config, "boundaries": {}}
    all_ensembles, representatives = {}, {}
    for boundary in ("periodic", "reflecting"):
        values = []
        tail_contrasts, tail_cv, tail_speed = [], [], []
        max_mass_error, max_p_error = 0, 0
        for k in range(seeds):
            rows, _, _ = run(width, height, steps, boundary, seed=2026 + k, block=block)
            write_csv(output / "data" / f"{boundary}_seed_{2026+k}.csv", rows)
            d = np.array([row["contrast"] for row in rows])
            values.append(d / d[0])
            tail = rows[-100:]
            tail_contrasts.append(np.mean([abs(row["contrast"]) for row in tail]))
            tail_cv.append(np.mean([row["density_cv"] for row in tail]))
            tail_speed.append(np.mean([row["rms_speed"] for row in tail]))
            max_mass_error = max(max_mass_error, max(abs(row["mass_error"]) for row in rows))
            max_p_error = max(max_p_error, max(row["momentum_error"] for row in rows))
        a = np.array(values)
        ensemble = [{"step": t, "signed_mean": float(a[:, t].mean()),
                     "signed_sd": float(a[:, t].std(ddof=1)),
                     "absolute_mean": float(np.abs(a[:, t]).mean()),
                     "absolute_sd": float(np.abs(a[:, t]).std(ddof=1))}
                    for t in range(steps + 1)]
        write_csv(output / "data" / f"{boundary}_ensemble.csv", ensemble)
        all_ensembles[boundary] = ensemble
        times = sorted(set((0, min(40, steps), min(120, steps), min(300, steps), steps)))
        representative = run(width, height, steps, boundary, seed=2026, block=block,
                             snapshot_times=times)
        representatives[boundary] = representative
        rows, snapshots, final = representative
        np.savez_compressed(output / "data" / f"{boundary}_states.npz",
                            **{f"t_{t}": a for t, a in snapshots.items()}, final=final)
        abs_mean = np.abs(a).mean(axis=0)
        summary["boundaries"][boundary] = {
            "mass_error_max": max_mass_error, "momentum_balance_error_max": max_p_error,
            "threshold_time": stable_threshold(abs_mean),
            "tail_absolute_contrast_mean": float(np.mean(tail_contrasts)),
            "tail_absolute_contrast_sd": float(np.std(tail_contrasts, ddof=1)),
            "tail_density_cv_mean": float(np.mean(tail_cv)),
            "tail_density_cv_sd": float(np.std(tail_cv, ddof=1)),
            "tail_rms_speed_mean": float(np.mean(tail_speed)),
            "tail_rms_speed_sd": float(np.std(tail_speed, ddof=1)),
            "representative_initial": rows[0], "representative_final": rows[-1],
        }
        print(f"{boundary}: {seeds} runs completed; conservation errors = 0", flush=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary, all_ensembles, representatives
