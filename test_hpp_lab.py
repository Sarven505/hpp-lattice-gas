import itertools

import numpy as np
import pytest

from hpp_lab import (RIGHT, UP, LEFT, DOWN, collide, stream, momentum,
                     initial_state, main)


def field(w, h, *particles):
    """Пустое поле с частицами; каждая задаётся как (x, y, направление)."""
    n = np.zeros((h, w, 4), dtype=bool)
    for x, y, k in particles:
        n[y, x, k] = True
    return n


def test_collision():
    # две встречные частицы разлетаются под прямым углом
    n = field(4, 4, (1, 1, RIGHT), (1, 1, LEFT), (2, 2, UP), (2, 2, DOWN))
    new = collide(n)
    assert list(new[1, 1]) == [False, True, False, True]
    assert list(new[2, 2]) == [True, False, True, False]
    # три частицы не меняются
    three = field(4, 4, (1, 1, RIGHT), (1, 1, LEFT), (1, 1, UP))
    assert (collide(three) == three).all()
    # для всех 16 состояний сохраняются число частиц и импульс
    for state in itertools.product([False, True], repeat=4):
        n = np.array(state).reshape(1, 1, 4)
        new = collide(n)
        assert new.sum() == n.sum()
        assert momentum(new) == momentum(n)


@pytest.mark.parametrize("boundary", ["periodic", "reflecting"])
def test_conservation(boundary):
    n = initial_state(1, 16, 12)
    total, p0 = n.sum(), momentum(n)
    for _ in range(100):
        n = stream(collide(n), boundary)
        assert n.sum() == total
        if boundary == "periodic":
            assert momentum(n) == p0


def test_single_particle_returns():
    w = 8
    n0 = field(w, 6, (2, 3, RIGHT))
    n = n0
    for _ in range(w):
        n = stream(collide(n), "periodic")
    assert (n == n0).all()
    n = n0
    for _ in range(2 * w):
        n = stream(collide(n), "reflecting")
    assert (n == n0).all()


def test_reflecting_wall():
    n = field(8, 6, (7, 3, RIGHT))
    new = stream(n, "reflecting")
    assert new.sum() == 1
    assert new[3, 7, LEFT]


def test_initial_state():
    a = initial_state(2026, 128, 96)
    b = initial_state(2026, 128, 96)
    assert (a == b).all()
    assert a[:, :64].mean() > a[:, 64:].mean() + 0.2


def test_bad_parameters():
    with pytest.raises(SystemExit):
        main(["--width", "7"])
    with pytest.raises(SystemExit):
        main(["--block", "7"])
