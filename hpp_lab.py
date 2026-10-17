import os

import numpy as np
import matplotlib.pyplot as plt

# параметры расчёта 
W, H = 128, 96        # ширина и высота поля (ширина чётная)
STEPS = 800           # число шагов по времени
P_LEFT = 0.45         # вероятность занятости канала, левая половина
P_RIGHT = 0.05        # вероятность занятости канала, правая половина
RUNS = 12             # сколько случайных начальных состояний берём
BLOCK = 8             # размер блока для усреднения плотности на картах
SHOW_TIMES = [0, 50, 150, 800]   # моменты времени для карт плотности
TAIL = 100            # по скольким последним шагам усредняем итог

# Четыре направления движения частицы в узле
RIGHT, UP, LEFT, DOWN = 0, 1, 2, 3
# Состояние поля: булев массив n[y, x, k]; True - в узле (x, y)
# есть частица, летящая в направлении k. Ось y направлена вверх.

BOUNDARIES = ["periodic", "reflecting"]
NAMES = {"periodic": "периодические границы",
         "reflecting": "отражающие границы"}


def initial_state(seed):
    """Случайное начальное заполнение: слева плотнее, чем справа."""
    rng = np.random.default_rng(seed)
    p = np.full((H, W, 1), P_RIGHT)
    p[:, :W // 2] = P_LEFT
    return rng.random((H, W, 4)) < p


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


def run(boundary, seed):
    """Один расчёт. Возвращает разность плотностей левой и правой половин
    на каждом шаге и снимки поля в моменты SHOW_TIMES."""
    n = initial_state(seed)
    total = n.sum()
    p0 = momentum(n)
    diff = []
    snapshots = {}
    for t in range(STEPS + 1):
        left = n[:, :W // 2].sum() / (H * W // 2)
        right = n[:, W // 2:].sum() / (H * W // 2)
        diff.append(left - right)
        if t in SHOW_TIMES:
            snapshots[t] = n.copy()
        # проверка законов сохранения
        assert n.sum() == total, "число частиц изменилось"
        if boundary == "periodic":
            assert momentum(n) == p0, "импульс изменился"
        n = stream(collide(n), boundary)
    return np.array(diff), snapshots


def block_density(n):
    """Плотность (частиц на узел), усреднённая по блокам BLOCK x BLOCK."""
    rho = n.sum(axis=2)
    return rho.reshape(H // BLOCK, BLOCK, W // BLOCK, BLOCK).mean(axis=(1, 3))


def maxima(curve, half_width=30):
    """Шаги, на которых кривая достигает положительного максимума
    (наибольшее значение среди соседних шагов)."""
    result = []
    for t in range(half_width, len(curve) - half_width):
        window = curve[t - half_width:t + half_width + 1]
        if curve[t] == window.max() and curve[t] > 0:
            result.append(t)
    return result


def main():
    os.makedirs("results", exist_ok=True)
    mean = {}         # средняя по запускам разность плотностей
    spread = {}       # разброс (стандартное отклонение) между запусками
    tail = {}         # средний |разность| на последних TAIL шагах, по запускам
    maps = {}         # снимки поля первого запуска

    for b in BOUNDARIES:
        curves = []
        for k in range(RUNS):
            diff, snaps = run(b, 2026 + k)
            curves.append(diff)
            if k == 0:
                maps[b] = snaps
        curves = np.array(curves)
        mean[b] = curves.mean(axis=0)
        spread[b] = curves.std(axis=0, ddof=1)
        tail[b] = np.abs(curves[:, -TAIL:]).mean(axis=1)
        print(f"{NAMES[b]}: {RUNS} расчётов выполнено, "
              "число частиц сохраняется")

    # ---------- результаты в консоль ----------
    print("\nСредний |разность плотностей| на последних", TAIL, "шагах")
    for b in BOUNDARIES:
        print(f"  {NAMES[b]}: {tail[b].mean():.3f} "
              f"± {tail[b].std(ddof=1):.3f} частиц/узел")
    print("\nМаксимумы средней разности плотностей: шаг (значение)")
    for b in BOUNDARIES:
        peaks = [f"{t} ({mean[b][t]:.2f})" for t in maxima(mean[b])]
        print(f"  {NAMES[b]}: " + ", ".join(peaks))

    # ---------- график 1: разность плотностей от времени ----------
    t = np.arange(STEPS + 1)
    colors = {"periodic": "tab:blue", "reflecting": "tab:red"}
    plt.figure(figsize=(9, 4.5))
    for b in BOUNDARIES:
        plt.plot(t, mean[b], color=colors[b], label=NAMES[b])
        plt.fill_between(t, mean[b] - spread[b], mean[b] + spread[b],
                         color=colors[b], alpha=0.2)
    plt.axhline(0, color="gray", linewidth=0.8)
    plt.xlabel("шаг t")
    plt.ylabel("ρ левой − ρ правой, частиц/узел")
    plt.title(f"Разность плотностей половин поля (среднее по {RUNS} запускам)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/difference.png", dpi=150)
    plt.close()

    # ---------- график 2: карты плотности ----------
    fig, axes = plt.subplots(2, len(SHOW_TIMES), figsize=(12, 5.5),
                             constrained_layout=True)
    for i, b in enumerate(BOUNDARIES):
        for j, tt in enumerate(SHOW_TIMES):
            ax = axes[i, j]
            im = ax.imshow(block_density(maps[b][tt]), origin="lower",
                           extent=(0, W, 0, H), vmin=0, vmax=2.4,
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


if __name__ == "__main__":
    main()
