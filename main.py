"""Запустить все эксперименты лабораторной: python main.py."""

from pathlib import Path

from hpp.experiments import study
from hpp.plots import study_plots


def main():
    output = Path(__file__).resolve().parent / "results"
    print("HPP: сравнение периодических и отражающих границ")
    print("128 × 96 узлов, 800 шагов, 12 начальных состояний для каждой границы")
    _, ensembles, examples = study(output)
    study_plots(output / "figures", ensembles, examples, block=8, seeds=12)
    print(f"Готово. Данные: {output / 'data'}")
    print(f"Графики: {output / 'figures'}")


if __name__ == "__main__":
    main()
