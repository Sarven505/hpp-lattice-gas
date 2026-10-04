"""Independent reference dynamics and theoretical invariants, not snapshots of implementation."""

import itertools

import numpy as np
import pytest

from hpp.experiments import run, stable_threshold
from hpp.model import (VELOCITIES, collide, fields, initialize, inverse_step,
                       momentum, step, stream, validate_state, wall_impulse)


def reference_step(a, boundary):
    """Slow scalar model used only as an independent small-grid oracle."""
    h, w, _ = a.shape
    out = np.zeros_like(a)
    for y in range(h):
        for x in range(w):
            occupied = {i for i in range(4) if a[y, x, i]}
            if occupied == {0, 2}: occupied = {1, 3}
            elif occupied == {1, 3}: occupied = {0, 2}
            for i in occupied:
                dx, dy = VELOCITIES[i]
                xx, yy = x + int(dx), y + int(dy)
                if boundary == "periodic":
                    xx, yy = xx % w, yy % h
                    j = i
                elif not (0 <= xx < w and 0 <= yy < h):
                    xx, yy, j = x, y, (i + 2) % 4
                else: j = i
                assert not out[yy, xx, j]
                out[yy, xx, j] = True
    return out


@pytest.mark.parametrize("bits", list(itertools.product((False, True), repeat=4)))
def test_all_local_collisions(bits):
    a = np.tile(np.array(bits), (2, 2, 1))
    b = collide(a)
    expected = tuple(not v for v in bits) if bits in ((1,0,1,0), (0,1,0,1)) else bits
    assert tuple(b[0, 0]) == expected
    assert int(a.sum()) == int(b.sum())
    assert np.array_equal(momentum(a), momentum(b))
    assert np.array_equal(collide(b), a)


@pytest.mark.parametrize("boundary", ["periodic", "reflecting"])
def test_against_independent_scalar_model(boundary):
    a = initialize(8, 6, 0.7, 0.3, 31)
    for _ in range(15):
        expected = reference_step(a, boundary)
        actual = step(a, boundary)
        assert np.array_equal(actual, expected)
        a = actual


@pytest.mark.parametrize("boundary", ["periodic", "reflecting"])
def test_exact_conservation_and_inverse(boundary):
    a = initialize(16, 12, 0.5, 0.2, 17)
    original = a.copy()
    m0, p0 = int(a.sum()), momentum(a)
    impulse = np.zeros(2, dtype=np.int64)
    for _ in range(100):
        c = collide(a)
        impulse += wall_impulse(c, boundary)
        a = stream(c, boundary)
        assert int(a.sum()) == m0
        assert np.array_equal(momentum(a), p0 + impulse)
    for _ in range(100): a = inverse_step(a, boundary)
    assert np.array_equal(a, original)


@pytest.mark.parametrize("boundary,period", [("periodic", 8), ("reflecting", 16)])
def test_single_particle_theoretical_period(boundary, period):
    a = np.zeros((6, 8, 4), dtype=bool)
    a[2, 2, 0] = True
    original = a.copy()
    for t in range(1, period + 1):
        a = step(a, boundary)
        assert int(a.sum()) == 1
        assert np.array_equal(a, original) == (t == period)


def test_wall_impulse_for_corner_particles():
    a = np.zeros((4, 4, 4), dtype=bool)
    a[0, 0, 2:4] = True
    assert np.array_equal(momentum(stream(a, "reflecting")) - momentum(a), (2, 2))


def test_fields_density_weighted_velocity_and_empty():
    a = np.zeros((2, 2, 4), dtype=bool)
    rho, u = fields(a, 2)
    assert np.isfinite(u).all() and rho[0,0] == 0 and not u.any()
    a[0,0,0] = True; a[1,1,0] = True; a[1,1,1] = True
    rho, u = fields(a, 2)
    assert rho[0,0] == 0.75
    assert np.allclose(u[0,0], (2/3, 1/3))


def test_uniform_initialization_binomial_density():
    # 65,536 nodes: statistical check with wide, explicitly stated tolerances.
    a = initialize(256, 256, 0.25, 0.25, 2026)
    rho = a.sum(axis=-1)
    assert abs(rho.mean() - 1.0) < 0.015
    assert abs(rho.var() - 0.75) < 0.025


def test_integer_empty_probability_does_not_truncate_fractional_probability():
    a = initialize(32, 24, 0.45, 0, 9)
    assert a[:, :16].any()
    assert not a[:, 16:].any()
    assert np.array_equal(a, initialize(32, 24, 0.45, 0.0, 9))


def test_reproducible_without_mutating_input():
    a = initialize(8, 8, seed=9)
    b = a.copy()
    step(a)
    assert np.array_equal(a, b)
    assert np.array_equal(a, initialize(8, 8, seed=9))


def test_bad_inputs():
    with pytest.raises(ValueError): initialize(7, 8)
    with pytest.raises(ValueError): initialize(8, 8, 1.1)
    with pytest.raises(ValueError): validate_state(np.zeros((4,4,4), dtype=int))
    with pytest.raises(ValueError): fields(initialize(8,8), 3)
    with pytest.raises(ValueError): step(initialize(8,8), "unknown")


def test_run_diagnostics_and_snapshot_consistency():
    rows, snapshots, final = run(16, 16, 20, "reflecting", block=4, snapshot_times=(0,20))
    assert len(rows) == 21
    assert all(r["mass_error"] == r["momentum_error"] == 0 for r in rows)
    assert np.array_equal(snapshots[20], final)


def test_threshold_requires_whole_window():
    assert stable_threshold([1,0,1,0,0,0], window=3) == 3
    assert stable_threshold([1,0,1,0], window=3) is None
