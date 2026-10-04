"""Synchronous HPP dynamics. Array axes are (y, x, direction); y points up."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

Boundary = Literal["periodic", "reflecting"]
E, N, W, S = range(4)
VELOCITIES = np.array(((1, 0), (0, 1), (-1, 0), (0, -1)), dtype=np.int64)
OPPOSITE = np.array((W, S, E, N))


def validate_state(state: np.ndarray) -> None:
    if state.ndim != 3 or state.shape[2] != 4 or min(state.shape[:2]) < 2:
        raise ValueError("State must have shape (height >= 2, width >= 2, 4).")
    if state.dtype != np.bool_:
        raise ValueError("Occupation numbers must have Boolean dtype.")


def collide(state: np.ndarray) -> np.ndarray:
    """Only an isolated head-on pair turns by 90 degrees; do not mutate input."""
    validate_state(state)
    horizontal = state[..., E] & state[..., W] & ~state[..., N] & ~state[..., S]
    vertical = state[..., N] & state[..., S] & ~state[..., E] & ~state[..., W]
    hit = horizontal | vertical
    result = state.copy()
    result[..., E][hit] = vertical[hit]
    result[..., W][hit] = vertical[hit]
    result[..., N][hit] = horizontal[hit]
    result[..., S][hit] = horizontal[hit]
    return result


def stream(state: np.ndarray, boundary: Boundary) -> np.ndarray:
    """Move to a neighbour; a wall reflects into the opposite source channel."""
    validate_state(state)
    if boundary == "periodic":
        result = np.empty_like(state)
        for i, (dx, dy) in enumerate(VELOCITIES):
            result[..., i] = np.roll(state[..., i], (int(dy), int(dx)), (0, 1))
        return result
    if boundary != "reflecting":
        raise ValueError("Boundary must be 'periodic' or 'reflecting'.")
    result = np.zeros_like(state)
    result[:, 1:, E] = state[:, :-1, E]
    result[:, 0, E] = state[:, 0, W]
    result[:, :-1, W] = state[:, 1:, W]
    result[:, -1, W] = state[:, -1, E]
    result[1:, :, N] = state[:-1, :, N]
    result[0, :, N] = state[0, :, S]
    result[:-1, :, S] = state[1:, :, S]
    result[-1, :, S] = state[-1, :, N]
    return result


def wall_impulse(post_collision: np.ndarray, boundary: Boundary) -> np.ndarray:
    """Impulse delivered by walls to the gas during the next streaming phase."""
    if boundary == "periodic":
        return np.zeros(2, dtype=np.int64)
    if boundary != "reflecting":
        raise ValueError("Unknown boundary.")
    a = post_collision
    return np.array((2 * (int(a[:, 0, W].sum()) - int(a[:, -1, E].sum())),
                     2 * (int(a[0, :, S].sum()) - int(a[-1, :, N].sum()))),
                    dtype=np.int64)


def step(state: np.ndarray, boundary: Boundary = "periodic") -> np.ndarray:
    return stream(collide(state), boundary)


def inverse_step(state: np.ndarray, boundary: Boundary = "periodic") -> np.ndarray:
    """Invert S C in the correct order: C S^-1 (a useful reversibility check)."""
    unstreamed = stream(state[..., OPPOSITE], boundary)[..., OPPOSITE]
    return collide(unstreamed)


def initialize(width: int = 128, height: int = 96, p_left: float = 0.45,
               p_right: float = 0.05, seed: int = 2026) -> np.ndarray:
    """Independent Bernoulli occupation of each channel, different in two halves."""
    if width < 2 or height < 2 or width % 2:
        raise ValueError("Width must be even and >= 2; height must be >= 2.")
    if not (0 <= p_left <= 1 and 0 <= p_right <= 1):
        raise ValueError("Occupation probabilities must be between 0 and 1.")
    probability = np.full((height, width, 1), p_right, dtype=float)
    probability[:, :width // 2] = p_left
    return np.random.default_rng(seed).random((height, width, 4)) < probability


def momentum(state: np.ndarray) -> np.ndarray:
    return state.sum(axis=(0, 1), dtype=np.int64) @ VELOCITIES


def fields(state: np.ndarray, block: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Block-average density and density-weighted velocity (not mean node velocity)."""
    h, w, _ = state.shape
    if block < 1 or h % block or w % block:
        raise ValueError("Positive block size must divide width and height.")
    population = state.reshape(h // block, block, w // block, block, 4).mean(axis=(1, 3))
    rho = population.sum(axis=-1)
    flux = population @ VELOCITIES
    velocity = np.divide(flux, rho[..., None], out=np.zeros_like(flux),
                         where=rho[..., None] > 0)
    return rho, velocity


@dataclass
class Diagnostics:
    step: int
    mass: int
    px: int
    py: int
    rho_left: float
    rho_right: float
    contrast: float
    density_cv: float
    rms_speed: float
    mass_error: int
    momentum_error: int


def measure(state: np.ndarray, t: int, block: int, mass0: int,
            momentum0: np.ndarray, cumulative_impulse: np.ndarray) -> Diagnostics:
    rho, velocity = fields(state, block)
    mid = state.shape[1] // 2
    left = float(state[:, :mid].sum()) / (state.shape[0] * mid)
    right = float(state[:, mid:].sum()) / (state.shape[0] * mid)
    mass = int(state.sum())
    p = momentum(state)
    cv = float(rho.std() / rho.mean()) if rho.mean() else 0.0
    rms = float(np.sqrt(np.mean(np.sum(velocity ** 2, axis=-1))))
    error = int(np.abs(p - momentum0 - cumulative_impulse).max())
    return Diagnostics(t, mass, int(p[0]), int(p[1]), left, right, left - right,
                       cv, rms, mass - mass0, error)
