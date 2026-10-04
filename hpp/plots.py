"""Static scientific figures and an optional GIF; no display server required."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .model import fields

COLORS = {"periodic": "#246b9d", "reflecting": "#c26635"}
LABELS = {"periodic": "Периодические границы", "reflecting": "Отражающие границы"}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 140, "savefig.dpi": 180})


def study_plots(output, ensembles, representatives, block=8, seeds=12):
    style()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True, constrained_layout=True)
    for boundary, rows in ensembles.items():
        t = np.array([r["step"] for r in rows])
        m, sd = (np.array([r[key] for r in rows]) for key in ("signed_mean", "signed_sd"))
        axes[0].plot(t, m, color=COLORS[boundary], label=LABELS[boundary])
        axes[0].fill_between(t, m-sd, m+sd, color=COLORS[boundary], alpha=0.15)
        axes[1].plot(t, [r["absolute_mean"] for r in rows], color=COLORS[boundary])
    axes[0].axhline(0, color="#9ca3af", lw=0.8)
    axes[0].set_ylabel(r"$(\rho_L-\rho_R)/\Delta\rho(0)$")
    axes[0].legend(fontsize=9)
    axes[0].set_title(f"Разность плотностей: среднее ± 1 SD по {seeds} начальным состояниям")
    axes[1].axhline(0.1, color="#667085", linestyle="--", label="Порог 0.1")
    axes[1].set_ylabel(r"$\langle |\Delta\rho/\Delta\rho(0)|\rangle$")
    axes[1].set_xlabel("Шаг моделирования t")
    axes[1].legend()
    for ax in axes: ax.grid(alpha=0.2)
    fig.savefig(output / "mixing.png")
    plt.close(fig)
    for boundary, (rows, snapshots, _) in representatives.items():
        times = sorted(set((min(snapshots), sorted(snapshots)[len(snapshots)//2], max(snapshots))))
        fig, axes = plt.subplots(2, len(times), figsize=(9, 5), constrained_layout=True, squeeze=False)
        for j, t in enumerate(times):
            state = snapshots[t]
            rho, u = fields(state, block)
            h, w, _ = state.shape
            extent = (0, w, 0, h)
            im = axes[0, j].imshow(rho, origin="lower", extent=extent,
                                    vmin=0, vmax=2.4, cmap="viridis", aspect="equal")
            speed = np.linalg.norm(u, axis=-1)
            im2 = axes[1, j].imshow(speed, origin="lower", extent=extent,
                                     vmin=0, vmax=1, cmap="magma", aspect="equal")
            y, x = np.mgrid[block/2:h:block, block/2:w:block]
            axes[1, j].quiver(x, y, u[..., 0], u[..., 1], color="white", scale=0.12,
                              angles="xy", scale_units="xy", width=0.006, headwidth=3)
            axes[0, j].set_title(f"t = {t}")
            axes[1, j].set_xlabel("x")
            for ax in axes[:, j]: ax.set_xticks((0, w//2, w)); ax.set_yticks((0, h//2, h))
        axes[0, 0].set_ylabel("Плотность: y")
        axes[1, 0].set_ylabel("Скорость: y")
        fig.colorbar(im, ax=list(axes[0]), shrink=0.75, label="Частиц / узел")
        fig.colorbar(im2, ax=list(axes[1]), shrink=0.75, label="Узлов / шаг")
        fig.suptitle(f"{LABELS[boundary]}; seed = 2026; усреднение {block} × {block}")
        fig.savefig(output / f"fields_{boundary}.png")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for boundary, (rows, _, _) in representatives.items():
        t = [r["step"] for r in rows]
        for ax, key in zip(axes, ("density_cv", "rms_speed")):
            ax.plot(t, [r[key] for r in rows], color=COLORS[boundary], label=LABELS[boundary])
            ax.grid(alpha=0.2); ax.set_xlabel("Шаг t")
    axes[0].set_ylabel(f"CV плотности по блокам {block} × {block}")
    axes[1].set_ylabel("RMS макроскопической скорости")
    axes[1].legend(fontsize=8)
    fig.savefig(output / "observables.png")
    plt.close(fig)


def run_plot(output, rows, snapshots, boundary, block):
    style()
    output = Path(output)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    t = [r["step"] for r in rows]
    axes[0].plot(t, [r["rho_left"] for r in rows], label="Левая половина")
    axes[0].plot(t, [r["rho_right"] for r in rows], label="Правая половина")
    axes[0].set_ylabel("Средняя плотность"); axes[0].legend()
    axes[1].plot(t, [r["rms_speed"] for r in rows]); axes[1].set_ylabel("RMS скорости")
    for ax in axes: ax.set_xlabel("Шаг t"); ax.grid(alpha=0.2)
    fig.savefig(output / "diagnostics.png"); plt.close(fig)
    final = snapshots[max(snapshots)]
    rho, u = fields(final, block)
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    im = ax.imshow(rho, origin="lower", vmin=0, vmax=4, cmap="viridis")
    ax.set_title(LABELS[boundary]); ax.set_xlabel("Блок x"); ax.set_ylabel("Блок y")
    fig.colorbar(im, ax=ax, label="Частиц / узел")
    fig.savefig(output / "density_final.png"); plt.close(fig)


def save_gif(path, snapshots, scale=4):
    """A compact overview of raw node density; fixed colour scale from 0 to 4."""
    palette = np.array([[247,249,252],[70,150,181],[41,97,154],[94,55,120],[244,191,70]], dtype=np.uint8)
    frames = []
    for state in snapshots.values():
        rgb = palette[state.sum(axis=-1, dtype=np.int64)]
        im = Image.fromarray(rgb).resize((state.shape[1]*scale, state.shape[0]*scale),
                                        Image.Resampling.NEAREST)
        frames.append(im)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=65, loop=0)
