"""
Лабораторная работа: решётчатый газ HPP.

В левой половине поля частиц больше, чем в правой. Смотрим, как газ
перераспределяется при периодических и при отражающих границах.

Запуск:  python hpp_lab.py            (все параметры - по умолчанию)
         python hpp_lab.py --gif      (ещё и анимация images/animation.gif)
         python hpp_lab.py -h         (справка)
Нужны библиотеки numpy, matplotlib и pillow (pip install -r requirements.txt).
Графики сохраняются в папку results.
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------- параметры расчёта ----------
P_LEFT = 0.45         # вероятность занятости канала, левая половина
P_RIGHT = 0.05        # вероятность занятости канала, правая половина
TAIL = 100            # по скольким последним шагам усредняем итог
FRAMES = 50           # число кадров в анимации

# Четыре направления движения частицы в узле
RIGHT, UP, LEFT, DOWN = 0, 1, 2, 3
# Состояние поля: булев массив n[y, x, k]; True - в узле (x, y)
# есть частица, летящая в направлении k. Ось y направлена вверх.

BOUNDARIES = ["periodic", "reflecting"]
NAMES = {"periodic": "периодические границы",
         "reflecting": "отражающие границы"}


def initial_state(seed, w, h):
    """Случайное начальное заполнение: слева плотнее, чем справа."""
    rng = np.random.default_rng(seed)
    p = np.full((h, w, 1), P_RIGHT)
    p[:, :w // 2] = P_LEFT
    return rng.random((h, w, 4)) < p


def collide(n):
    """Столкновения. Две частицы, летящие навстречу друг другу, а остальные
    два канала пустые, разлетаются под прямым углом. Всё остальное
    остаётся как было."""
    r, u, l, d = n[..., RIGHT], n[..., UP], n[..., LEFT], n[..., DOWN]
    horizontal = r & l & ~u & ~d      # пара вправо + влево
    vertical = u & d & ~r & ~l        # пара вверх + вниз
    new = n.copy()
    new[horizontal] = [False, True, False, True]    # стала вертикальной
    new[vertical] = [True, False, True, False]      # стала горизонтальной
    return new


def stream(n, boundary):
    """Перенос: каждая частица смещается на один узел в своём направлении."""
    new = np.zeros_like(n)
    if boundary == "periodic":
        # вышла за край - появилась с противоположной стороны
        new[..., RIGHT] = np.roll(n[..., RIGHT], 1, axis=1)
        new[..., LEFT] = np.roll(n[..., LEFT], -1, axis=1)
        new[..., UP] = np.roll(n[..., UP], 1, axis=0)
        new[..., DOWN] = np.roll(n[..., DOWN], -1, axis=0)
    else:
        # обычное смещение на один узел
        new[:, 1:, RIGHT] = n[:, :-1, RIGHT]
        new[:, :-1, LEFT] = n[:, 1:, LEFT]
        new[1:, :, UP] = n[:-1, :, UP]
        new[:-1, :, DOWN] = n[1:, :, DOWN]
        # у стенки частица остаётся в том же узле и летит обратно
        new[:, -1, LEFT] = n[:, -1, RIGHT]
        new[:, 0, RIGHT] = n[:, 0, LEFT]
        new[-1, :, DOWN] = n[-1, :, UP]
        new[0, :, UP] = n[0, :, DOWN]
    return new


def momentum(n):
    """Суммарный импульс всех частиц (px, py)."""
    px = n[..., RIGHT].sum() - n[..., LEFT].sum()
    py = n[..., UP].sum() - n[..., DOWN].sum()
    return int(px), int(py)


def run(boundary, seed, w, h, steps, show_times):
    """Один расчёт. Возвращает разность плотностей левой и правой половин
    на каждом шаге и снимки поля в моменты show_times."""
    n = initial_state(seed, w, h)
    total = n.sum()
    p0 = momentum(n)
    diff = []
    snapshots = {}
    for t in range(steps + 1):
        left = n[:, :w // 2].sum() / (h * w // 2)
        right = n[:, w // 2:].sum() / (h * w // 2)
        diff.append(left - right)
        if t in show_times:
            snapshots[t] = n.copy()
        # проверка законов сохранения
        assert n.sum() == total, "число частиц изменилось"
        if boundary == "periodic":
            assert momentum(n) == p0, "импульс изменился"
        n = stream(collide(n), boundary)
    return np.array(diff), snapshots


def block_density(n, block):
    """Плотность (частиц на узел), усреднённая по блокам block x block."""
    h, w = n.shape[:2]
    rho = n.sum(axis=2)
    return rho.reshape(h // block, block, w // block, block).mean(axis=(1, 3))


def maxima(curve, half_width=30):
    """Шаги, на которых кривая достигает положительного максимума
    (наибольшее значение среди соседних шагов)."""
    result = []
    for t in range(half_width, len(curve) - half_width):
        window = curve[t - half_width:t + half_width + 1]
        if curve[t] == window.max() and curve[t] > 0:
            result.append(t)
    return result


def make_gif(maps, frame_times, w, h, block):
    """Анимация первого запуска: слева периодические границы,
    справа отражающие."""
    os.makedirs("images", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.3), constrained_layout=True)
    images = []
    for ax, b in zip(axes, BOUNDARIES):
        rho = block_density(maps[b][frame_times[0]], block)
        images.append(ax.imshow(rho, origin="lower", extent=(0, w, 0, h),
                                vmin=0, vmax=2.4, cmap="viridis"))
        ax.set_title(NAMES[b])
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(images[0], ax=axes, shrink=0.8,
                 label="плотность, частиц/узел")
    title = fig.suptitle("")

    def update(i):
        t = frame_times[i]
        for im, b in zip(images, BOUNDARIES):
            im.set_data(block_density(maps[b][t], block))
        title.set_text(f"t = {t}")

    anim = FuncAnimation(fig, update, frames=len(frame_times))
    anim.save("images/animation.gif", writer=PillowWriter(fps=10), dpi=80)
    plt.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Решётчатый газ HPP: периодические и отражающие границы.",
        add_help=False)
    parser.add_argument("-h", "--help", action="help",
                        help="показать справку и выйти")
    parser.add_argument("--width", type=int, default=128,
                        help="ширина поля, чётная (128)")
    parser.add_argument("--height", type=int, default=96,
                        help="высота поля (96)")
    parser.add_argument("--steps", type=int, default=800,
                        help="число шагов по времени (800)")
    parser.add_argument("--runs", type=int, default=12,
                        help="число случайных начальных состояний, не меньше 2 (12)")
    parser.add_argument("--block", type=int, default=8,
                        help="размер блока для карт плотности, делит ширину "
                             "и высоту (8)")
    parser.add_argument("--gif", action="store_true",
                        help="сделать анимацию images/animation.gif")
    args = parser.parse_args(argv)

    w, h, steps, runs, block = (args.width, args.height, args.steps,
                                args.runs, args.block)
    if w < 2 or w % 2 != 0:
        parser.error("ширина должна быть чётной и не меньше 2")
    if h < 2:
        parser.error("высота должна быть не меньше 2")
    if steps < 1:
        parser.error("число шагов должно быть не меньше 1")
    if runs < 2:
        parser.error("число запусков должно быть не меньше 2")
    if block < 1 or w % block != 0 or h % block != 0:
        parser.error("блок должен быть натуральным и делить ширину и высоту")

    show_times = [0, steps // 16, 3 * steps // 16, steps]
    frame_times = [int(t) for t in np.linspace(0, steps, FRAMES).round()]
    first_times = sorted(set(show_times + frame_times)) if args.gif \
        else show_times
    tail_len = min(TAIL, steps)

    os.makedirs("results", exist_ok=True)
    mean = {}         # средняя по запускам разность плотностей
    spread = {}       # разброс (стандартное отклонение) между запусками
    tail = {}         # средний |разность| на последних шагах, по запускам
    maps = {}         # снимки поля первого запуска

    for b in BOUNDARIES:
        curves = []
        for k in range(runs):
            times = first_times if k == 0 else show_times
            diff, snaps = run(b, 2026 + k, w, h, steps, times)
            curves.append(diff)
            if k == 0:
                maps[b] = snaps
        curves = np.array(curves)
        mean[b] = curves.mean(axis=0)
        spread[b] = curves.std(axis=0, ddof=1)
        tail[b] = np.abs(curves[:, -tail_len:]).mean(axis=1)
        print(f"{NAMES[b]}: {runs} расчётов выполнено, "
              "число частиц сохраняется")

    # ---------- результаты в консоль ----------
    print("\nСредний |разность плотностей| на последних", tail_len, "шагах")
    for b in BOUNDARIES:
        print(f"  {NAMES[b]}: {tail[b].mean():.3f} "
              f"± {tail[b].std(ddof=1):.3f} частиц/узел")
    print("\nМаксимумы средней разности плотностей: шаг (значение)")
    for b in BOUNDARIES:
        peaks = [f"{t} ({mean[b][t]:.2f})" for t in maxima(mean[b])]
        print(f"  {NAMES[b]}: " + ", ".join(peaks))

    # ---------- график 1: разность плотностей от времени ----------
    t = np.arange(steps + 1)
    colors = {"periodic": "tab:blue", "reflecting": "tab:red"}
    plt.figure(figsize=(9, 4.5))
    for b in BOUNDARIES:
        plt.plot(t, mean[b], color=colors[b], label=NAMES[b])
        plt.fill_between(t, mean[b] - spread[b], mean[b] + spread[b],
                         color=colors[b], alpha=0.2)
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.xlabel("шаг t")
    plt.ylabel("ρ левой − ρ правой, частиц/узел")
    plt.title(f"Разность плотностей половин поля (среднее по {runs} запускам)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/difference.png", dpi=150)
    plt.close()

    # ---------- график 2: карты плотности ----------
    fig, axes = plt.subplots(2, len(show_times), figsize=(12, 5.5),
                             constrained_layout=True)
    for i, b in enumerate(BOUNDARIES):
        for j, tt in enumerate(show_times):
            ax = axes[i, j]
            im = ax.imshow(block_density(maps[b][tt], block), origin="lower",
                           extent=(0, w, 0, h), vmin=0, vmax=2.4,
                           cmap="viridis")
            ax.set_title(f"t = {tt}")
            if j == 0:
                ax.set_ylabel(NAMES[b] + "\ny")
            if i == 1:
                ax.set_xlabel("x")
    fig.colorbar(im, ax=axes, shrink=0.8, label="плотность, частиц/узел")
    plt.savefig("results/density_maps.png", dpi=150)
    plt.close()
    print("\nГрафики сохранены в папку results")

    if args.gif:
        make_gif(maps, frame_times, w, h, block)
        print("Анимация сохранена: images/animation.gif")


if __name__ == "__main__":
    main()
