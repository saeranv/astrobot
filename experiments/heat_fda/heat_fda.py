"""
Derivation:
-q / x = Q
q = W/m2; Q = W/m3

# Expand
dq/dx = p C dT/dt
d2T/dx2 = p C dT/dt
d[-k dT/dx] = p C dT/dt

# Discretize in x and t dims
-k ((T3 - T2)/dx - (T2 - T1)/dx) / dx = p C (T2_t1 - T2_t0)/dt
-k (T3 - 2T2 + T1)/dx^2 = p C (T2_t1 - T2_t0)/dt

# Solve for T2_t1
T2_t1 = (-k dt / p C) (T3 - 2T2 + T1)/dx^2 + T2_t0

Refs:
https://scipython.com/book/chapter-7-matplotlib/examples/the-two-dimensional-diffusion-equation/
"Finite difference for heat equation in matrix form" Qiqi Wang, https://www.youtube.com/watch?v=YWRLnUDKBSE
"""

import numpy as np
import matplotlib.pyplot as plt

DX = 0.1
DX2 = DX * DX

def _diffusivity(C, p, k):
    """
    diffusivity = W/m2-K / J/kg-K * kg/m3
                = mW / J = mW / W*s
                = m/s
    """
    return k / (p * C) * 1000 # 4 mm/s

def _max_time_step(alpha, dx2):
    """Calculate maximum time step."""
    return 1 / (2 * alpha) * (np.square(dx2) / (dx2 + dx2))

def oned_loop(dimt=4, dimx=1.5, C=420, p=8, k=13.44):
    """
    1D Heat equation with for loop.

    Args:
        dimx = Number of nodes in x dimension spaced by 1mm.
        C = Specific heat capacity (J/kg-K)
        p = kg/m3
        k = W/m2-K

    Returns:
        2D matrix storing time, x dim.
    """

    # Set initial parameters
    t_lo, t_hi = 300, 700
    alpha = _diffusivity(C, p, k)  # 4 mm/
    dimx = int(dimx / DX)

    # Set up material matrix
    ux = np.ones((dimt, dimx, 1)) * t_lo
    ux[:, dimx // 2, :] = t_hi  # heat source

    dt = _max_time_step(alpha, DX2)

    for t in range(dimt-1):
        for i in range(1, dimx-1):
            # Central difference in 1D:
            #   T2_t1 = (-k dt / p C) (T3 - 2T2 + T1)/dx^2 + T2_t0
            ux[t + 1, i] = \
                (alpha * dt) * ((ux[t, i - 1] - (2 *  ux[t, i]) + ux[t, i + 1]) / DX2) + ux[t, i]

    return ux

def oned_mtx(dimt=4, dimx=1.5, C=420, p=8, k=13.44):
    """
    1D Heat equation with matrix calcs.

    Args:
        dimx = Number of nodes in x dimension.
        C = Specific heat capacity (J/kg-K)
        p = kg/m3
        k = W/m2-K
    """

    t_lo, t_hi = 300, 700
    alpha = _diffusivity(C, p, k)  # 4 mm/s
    dimx = int(dimx / DX)

    uxy = np.ones((dimt, dimx, 1)) * t_lo
    uxy[:, dimx // 2, :] = t_hi

    dt = _max_time_step(alpha, DX2)

    # A * v = b
    # A: matrix of constants
    # main diagonal: -2dt / dx2  (for T2)
    # lower diagonal: dt / dx2  (for T3)
    # upper diagonal: dt / dx2 (for T1)
    A = np.zeros((dimx, dimx))
    A += np.diag(v=np.ones((dimx)) * (-2 * dt) / DX2, k=0)
    A += np.diag(v=np.ones((dimx-1)) * dt / DX2, k=-1)
    A += np.diag(v=np.ones((dimx-1)) * dt / DX2, k=1)

    for t in range(dimt-1):
        uxy[t + 1, :, :] = (A @ uxy[t, :, :]) + uxy[t, :, :]  # (dimx x dimx) @ (dimx, 1) = dimx x 1

    return uxy


def twod_loop(dimt=201, dimx=4.1, dimy=4.1, C=420, p=8, k=13.44):
    """
    2D Heat equation with for loop.

    Args:
        dimx = Number of nodes in x dimension spaced by 1mm.
        dimy = Number of nodes in y dimension spaced by 1mm.
        C = Specific heat capacity (J/kg-K)
        p = kg/m3
        k = W/m2-K

    Returns:
        3D matrix storing time, x y dim.
    """

    # Set initial parameters
    t_lo, t_hi = 300, 700
    alpha = _diffusivity(C, p, k)  # 4 mm/
    dimx, dimy = int(dimx/DX), int(dimy/DX)

    # Set up material matrix
    uxy = np.ones((dimt, dimx, dimy)) * t_lo
    dt = _max_time_step(alpha, DX2)
    r, cx, cy = 1, 2, 2
    xx, yy = np.meshgrid(np.arange(dimx), np.arange(dimy))
    rad_mtx = (cx - xx * DX) ** 2 + (cy - yy * DX) ** 2
    uxy[0, rad_mtx < r * r] = t_hi

    for t in range(dimt-1):
        for ix in range(1, dimx-1):
            for iy in range(1, dimy-1):
                # Central difference in 2d:
                #   T2_t1 = (Tx3 + Ty3 - 4T2 + Tx1 + Ty1) / dx^2 + T2_t0
                ux = uxy[t, ix + 1, iy] - (2 *  uxy[t, ix, iy]) + uxy[t, ix - 1, iy]
                uy = uxy[t, ix, iy + 1] - (2 *  uxy[t, ix, iy]) + uxy[t, ix, iy - 1]
                uxy[t + 1, ix, iy] = \
                    (alpha * dt) * ((ux + uy) / DX2) + uxy[t, ix, iy]

    return uxy


def oned_steady_state(dimx=1.5, T0=100, TL=0, k=13.44):
    """
    1D steady heat conduction.

    Args:
        dimx = Number of nodes in x dimension spaced by 1mm.
        k = W/m2-K
        T0 = Temperature at 0th boundary condition.
        TL = Temeprature at Lth boundary condition

    Returns:
        2D matrix storing x dim.
    """

    # Set initial parameters
    print('# of nodes: {}/{} = {}'.format(dimx, DX, int(np.round(dimx/DX, 0))))
    dimx = int(np.round(dimx/DX, 0))
    # Set up material matrix
    # ux = np.ones((dimx, 1)) * t_lo
    # ux[:, 0, 1: -1] = t_hi

    # From pg. 325 of Cengel
    # k(Tx3 - 2*T2 + Tx1) / dx2 + k(Ty3 - 2*T2 + Ty1) / dy2 = 0
    # k(Tx3 + Ty3 - 4*T2 + Tx1 + Ty1) / dx2 = 0

    # A: matrix of constants
    # main diagonal: -2k / dx2  (for T2)
    # lower diagonal: k / dx2  (for T3)
    # upper diagonal: k / dx2 (for T1)
    A = np.zeros((dimx, dimx))
    A = np.diag(v=np.ones((dimx)) * -2 * k / DX2, k=0)
    A += np.diag(v=np.ones((dimx-1)) * k / DX2, k=-1)
    A += np.diag(v=np.ones((dimx-1)) * k / DX2, k=1)

    # Add boundary
    y = np.ones((dimx, 1))
    y[ 0, 0] = -k * T0 / DX2
    y[-1, 0] = -k * TL / DX2
    Ainv = np.linalg.pinv(A)
    ux = Ainv @ y

    return ux

if __name__ == "__main__":

    oned_loop()
    oned_mtx()
    twod_loop()







