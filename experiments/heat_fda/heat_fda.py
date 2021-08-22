"""
Derivation:
-dq / dx = Q
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

# Process is:
# Solve heat conducted: through volume: T1_t0, T2_t0, T3_t0 of boundary (must be some temp diff)
# Output is W. Add Qsol and Qir (if needed) = W.
# Multiply W by (dt / (p C)) and add T2_t0 = T2_t1

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
            # T2_t1 = (-k dt / p C) (T3 - 2T2 + T1)/dx^2 + T2_t0
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

    # Define spatial matrix of initial temperatures
    uxy = np.ones((dimt, dimx, 1)) * t_lo
    uxy[:, dimx // 2, :] = t_hi

    dt = _max_time_step(alpha, DX2)

    # A * v = b
    # A: adjacency matrix of constants N x N
    # v: N-vector of temps at timestep t
    # b: N-vector of delta temps after finite difference transformation
    # Note: Add b to v to get temps at timestep t + 1)
    # main diagonal: -2dt / dx2  (for T2)
    # lower diagonal: dt / dx2  (for T3)
    # upper diagonal: dt / dx2 (for T1)
    A = np.zeros((dimx, dimx))
    A += np.diag(v=np.ones((dimx)) * (-2 * dt) / DX2, k=0)
    A += np.diag(v=np.ones((dimx-1)) * dt / DX2, k=-1)
    A += np.diag(v=np.ones((dimx-1)) * dt / DX2, k=1)

    for t in range(dimt-1):
        # T2_t1 = (-k dt / p C) (T3 - 2T2 + T1)/dx^2 + T2_t0
        # (dimx x dimx) @ (dimx, 1) = dimx x 1
        uxy[t + 1, :, :] = (A @ uxy[t, :, :]) + uxy[t, :, :]

    return uxy


def twod_loop(dimt=201, dimx=4.1, dimr=1, C=420, p=8, k=13.44):
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
    dimx = dimy = int(dimx/DX)

    # Set up material matrix
    uxy = np.ones((dimt, dimx, dimy)) * t_lo
    dt = _max_time_step(alpha, DX2)
    r, cx, cy = dimr, int((dimx * DX)//2), int((dimy * DX)//2)
    xx, yy = np.meshgrid(np.arange(dimx), np.arange(dimy))
    rad_mtx = (cx - xx * DX) ** 2 + (cy - yy * DX) ** 2
    uxy[0, rad_mtx < r * r] = t_hi

    for t in range(dimt-1):
        for ix in range(1, dimx-1):
            for iy in range(1, dimy-1):
                # Central difference in 2d:
                #   T2_t1 = (-k dt / p C) * (Tx3 + Ty3 - 4T2 + Tx1 + Ty1) / dx^2 + T2_t0
                ux = uxy[t, ix + 1, iy] - (2 *  uxy[t, ix, iy]) + uxy[t, ix - 1, iy]
                uy = uxy[t, ix, iy + 1] - (2 *  uxy[t, ix, iy]) + uxy[t, ix, iy - 1]
                uxy[t + 1, ix, iy] = \
                    (alpha * dt) * ((ux + uy) / DX2) + uxy[t, ix, iy]

    return uxy


def twod_mtx(dimt=201, dimx=4.1, dimr=1, C=420, p=8, k=13.44):
    """
    2D Heat equation with matrix.

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
    dimx = dimy = int(dimx/DX)

    # Set up material matrix
    uxy = np.ones((dimt, dimx, dimy)) * t_lo
    dt = _max_time_step(alpha, DX2)
    r, cx, cy = dimr, int((dimx * DX)//2), int((dimy * DX)//2)
    xx, yy = np.meshgrid(np.arange(dimx), np.arange(dimy))
    rad_mtx = (cx - xx * DX) ** 2 + (cy - yy * DX) ** 2
    uxy[0, rad_mtx < r * r] = t_hi

    for t in range(dimt-1):
        # Central difference in 2d:
        #   T2_t1 = (-k dt / p C) * (Tx3 + Ty3 - 4T2 + Tx1 + Ty1) / dx^2 + T2_t0
        # We shift a dimx-1 x dimx-1 square so that the following relation occurs:
        #   Tx3 - 2Tx2 + Tx1
        ux = uxy[t, 2:, 1:-1] - (2 * uxy[t, 1:-1, 1:-1]) + uxy[t, :-2, 1:-1]
        # We shift a dimx-1 x dimx-1 square so that the following reln occurs:
        #   Ty3 - 2Ty2 + Ty1
        uy = uxy[t, 1:-1, 2:] - (2 * uxy[t, 1:-1, 1:-1]) + uxy[t, 1:-1, :-2]
        # Add everything up
        uxy[t + 1, 1:-1, 1:-1] = (alpha * dt) * ((ux + uy) / DX2) + uxy[t, 1:-1, 1:-1]

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
        1D matrix storing temperatures.
    """

    # Set initial parameters
    print('# of nodes: {}/{} = {}'.format(dimx, DX, int(np.round(dimx/DX, 0))))
    dimx = int(np.round(dimx/DX, 0))

    # A: matrix of constants
    # Based on equations on pg. 325 of Cengel
    # k(Tx3 - 2*T2 + Tx1) / dx2 + k(Ty3 - 2*T2 + Ty1) / dy2 = 0
    # k(Tx3 + Ty3 - 4*T2 + Tx1 + Ty1) / dx2 = 0

    # Note: can remove k/DX2 since everything is constant. It cancels out.
    # but nice to have now in case k/DX changes.

    A = np.zeros((dimx, dimx))
    # main diagonal: -2k / dx2  (for T2s)
    A += np.diag(v=np.ones((dimx)) * -2 * k / DX2, k=0)
    # lower diagonal: k / dx2  (for T3s)
    A += np.diag(v=np.ones((dimx-1)) * k / DX2, k=-1)
    # upper diagonal: k / dx2 (for T1s)
    A += np.diag(v=np.ones((dimx-1)) * k / DX2, k=1)

    # Add boundary
    y = np.zeros((dimx, 1))
    y[ 0, 0] = -k * T0 / DX2
    y[-1, 0] = -k * TL / DX2
    Ainv = np.linalg.pinv(A)
    ux = Ainv @ y

    return ux


def _2d_adj_mtx(dimx, dimy):
    """Construct adjacency matrix capturing 2D nodes.

    For the following lattice of nodes with dimx=4, dimy=2:

        N1 N2 N3 N4
        N5 N6 N7 N8

    The following matrix will be constructed:

            N1  N2  N3  N4  N5  N6  N7  N8
        N1  -4   1   0   0   1   0   0   0  = -Tup - Tleft
        N2   1  -4   1   0   0   1   0   0  = -Tup
        N3   0   1  -4   1   0   0   1   0  = -Tup
        N4   0   0   1  -4   0   0   0   1  = -Tup - Tright
        -----------------------------------
        N5   1   0   0   0  -4   1   0   0  = -Tdown - Tleft
        N6   0   1   0   0   1  -4   1   0  = -Tdown
        N7   0   0   1   0   0   1  -4   1  = -Tdown
        N8   0   0   0   1   0   0   1  -4  = -Tdown - Tright

    Note that between N4 and N5 the boundary conditions are modified
    to reflect there is no spatial adjacency between them.
    """
    dimn = dimx * dimy
    A = np.zeros((dimn, dimn)).astype(np.float64)

    # main diagonal for current node
    A += np.diag(v=np.ones(dimn) * -4, k=0)

    # Set left/right diagonals
    # lower diagonal 1 for left node
    A += np.diag(v=np.ones(dimn-1), k=-1)
    # upper diagonal 1 for right node
    A += np.diag(v=np.ones(dimn-1), k=1)

    # Set up/down diagonals offset by number of nodes in dimx
    # lower diagonal 2 for up node (offset by -dimx)
    A += np.diag(v=np.ones(dimn-dimx), k=-dimx)
    # upper diagonal 2 for down node (offset by dimx)
    A += np.diag(v=np.ones(dimn-dimx), k=dimx)

    # Remove the left/right boundary nodes at the row start/end
    i = np.arange(1, dimy).astype(int)
    ri = i * dimx
    A[ri - 1, ri] = 0 # remove right-most value for top row
    A[ri, ri - 1] = 0 # remove left-most value for bottom row

    return A


def twod_steady_state(dimx, dimy, bcs, k=13.44, qgens=None):
    """
    2D steady heat conduction.
    #   T2_t1 = (Tx3 + Ty3 - 4T2 + Tx1 + Ty1) / dx^2 + T2_t0

    Args:
        k: W/m2-K
        bcs: top, bottom, left, right boundary conditions as
            specific temperatures.
    Returns:
        2D matrix storing temperatures.
    """

    # Unpack bc temperatures
    Tup, Tdown, Tleft, Tright = bcs
    qgens = [] if not qgens else qgens

    # A: matrix of constants, but built out for 2 dimensions
    # Based on equations on pg. 326 of Cengel
    # k(Tx3 - 2*T2 + Tx1) / dx2 + k(Ty3 - 2*T2 + Ty1) / dy2 = 0
    # k(Tx3 + Ty3 - 4*T2 + Tx1 + Ty1) / dx2 = 0

    A = _2d_adj_mtx(dimx, dimy)

    # Add constant temps from boundary
    y = np.zeros((dimx * dimy, 1))

    # Add left/right bc to left/right-most nodes
    # get row idxs for where rows end, add -Tleft -Tright
    row_bc_idxs = np.arange(0, dimx * dimy, dimx).astype(int)
    # Add right most specific temp bc
    y[row_bc_idxs - 1] += -Tright
    # Add left most specific temp bc
    y[row_bc_idxs] += -Tleft

    # Add up/down bc to upper/lower-most nodes
    row_bc_idxs = np.arange(dimx).astype(int)  # first dimx nodes
    y[row_bc_idxs] += -Tup
    row_bc_idxs = (dimx * dimy) - 1 - row_bc_idxs  # last dimx nodes
    y[row_bc_idxs] += -Tdown

    # Add a heat source
    for ci, ri, qgen in qgens:
       A_idx = (ri * dimx) + ci
       y[A_idx] += -(qgen * DX2 / k)  # add heat source and convert to C

    # multiply by k/DX2 (optional, comment when debugging)
    A *= k / DX2
    y *= k / DX2

    # Solve
    Ainv = np.linalg.pinv(A)
    uxy = Ainv @ y

    # For debugging
    #xx = np.hstack([A, y])
    #print(xx.astype(int))

    return uxy.reshape(dimy, dimx)

if __name__ == "__main__":

    oned_loop()
    oned_mtx()
    twod_loop()
    twod_mtx()
    oned_steady_state()








