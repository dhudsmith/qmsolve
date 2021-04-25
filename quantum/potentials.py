import numpy as np
from typing import List, Tuple


def hard_disk(x, y, r=1.0, center=(0, 0), scale=1e5):
    """
    A circular barrier
    """
    xc, yc = center
    v_grid = np.zeros((len(x), len(y)))
    barrier_cond = np.add.outer((y - yc) ** 2, (x - xc) ** 2) <= r ** 2
    v_grid[barrier_cond] = scale

    return v_grid


def multiple_hard_disks(x, y, rs: List[float], centers: List[Tuple[float, float]]):
    """
    Multiple circular barriers
    """
    v = sum(hard_disk(x, y, r, c) for r, c in zip(rs, centers))
    return v


def ring_with_gaps(x, y, radius, width=0.1, height=1e6, gap_angle=30 * np.pi / 180, num_gaps=5, x_center=0.5,
                   y_center=0.5):
    """
    A ring with gaps
    """
    V = np.zeros_like(np.outer(y, x))

    # radius
    dsq = np.add.outer((y - y_center) ** 2, (x - x_center) ** 2)
    cond_circle = (dsq < (radius + width / 2) ** 2) & (dsq > (radius - width / 2) ** 2)

    # angle
    phis = np.arctan(np.divide.outer((y - y_center) + 1e-6, (x - x_center) + 1e-6))
    phis[:, x < x_center] = phis[:, x < x_center] + np.pi
    phis = phis + np.pi / 2

    cond_angle = np.zeros_like(phis, dtype=np.bool8)
    for i in range(num_gaps):
        start_angle = i * 2 * np.pi / num_gaps
        cond_gap = ((phis > start_angle) & (phis <= (start_angle + gap_angle)))
        cond_angle = cond_angle | cond_gap

    V[cond_circle & ~cond_angle] = height
    return V
