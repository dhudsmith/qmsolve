import numpy as np


def coherent_state_2d(x, y, p=(0, 0), xy0=(0, 0), w=(1, 1)):
    """
    2D Harmonic oscillator coherent state for initial state wave function
    """
    px, py = p
    x0, y0 = xy0
    wx, wy = w

    psi = (wx * wy) ** 0.25 * np.outer(
        np.exp(1.j * py * y - 1 / 2 * wy * (y - y0) ** 2),
        np.exp(1.j * px * x - 1 / 2 * wx * (x - x0) ** 2)
    )

    return psi
