"""python -m hpp: simulate one initial condition or reproduce the report study."""

import argparse
import json
from pathlib import Path

import numpy as np

from .experiments import run, study, write_csv
from .plots import run_plot, save_gif, study_plots


def positive(value):
    n = int(value)
    if n < 1: raise argparse.ArgumentTypeError("Значение должно быть положительным.")
    return n


def main():
    parser = argparse.ArgumentParser(description="Решётчатый газ HPP: плотность, скорость, законы сохранения.")
    sub = parser.add_subparsers(dest="command", required=True)
    sim = sub.add_parser("simulate", help="Один вычислительный эксперимент")
    sim.add_argument("--boundary", choices=("periodic", "reflecting"), default="periodic")
    sim.add_argument("--p-left", type=float, default=0.45)
    sim.add_argument("--p-right", type=float, default=0.05)
    sim.add_argument("--seed", type=int, default=2026)
    sim.add_argument("--gif", action="store_true", help="Сохранить анимацию плотности")
    exp = sub.add_parser("study", help="Сравнить границы по ансамблю начальных условий")
    exp.add_argument("--seeds", type=positive, default=12)
    for p in (sim, exp):
        p.add_argument("--width", type=positive, default=128)
        p.add_argument("--height", type=positive, default=96)
        p.add_argument("--steps", type=positive, default=800)
        p.add_argument("--block", type=positive, default=8)
        p.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.width % 2 or args.width < 2 or args.height < 2:
        parser.error("Ширина должна быть чётной и >= 2; высота должна быть >= 2.")
    if args.width % args.block or args.height % args.block:
        parser.error("Размер блока должен делить ширину и высоту.")
    try:
        if args.command == "study":
            _, ensembles, representatives = study(args.output, args.width, args.height,
                                                  args.steps, args.seeds, args.block)
            study_plots(args.output / "figures", ensembles, representatives, args.block, args.seeds)
        else:
            if not 0 <= args.p_left <= 1 or not 0 <= args.p_right <= 1:
                parser.error("Вероятности должны принадлежать [0, 1].")
            args.output.mkdir(parents=True, exist_ok=True)
            stride = max(1, args.steps // 100)
            times = sorted(set(range(0, args.steps+1, stride)) | {args.steps})
            rows, snapshots, final = run(args.width, args.height, args.steps, args.boundary,
                                         args.p_left, args.p_right, args.seed, args.block, times)
            write_csv(args.output / "metrics.csv", rows)
            np.savez_compressed(args.output / "states.npz", initial=snapshots[0], final=final)
            config = {key: str(value) if isinstance(value, Path) else value
                      for key, value in vars(args).items()}
            (args.output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))
            run_plot(args.output, rows, snapshots, args.boundary, args.block)
            if args.gif: save_gif(args.output / "density.gif", snapshots)
            print(f"Результаты: {args.output.resolve()}\nЧастиц: {rows[-1]['mass']}; ошибки балансов = 0")
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
