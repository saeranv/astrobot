import numpy as np
from astrobot import mtx_util


PNTMTX_XY = np.array(
    [[-6.,  6.,  6., -6.],
     [-3., -3.,  3.,  3.],
     [ 0.,  0.,  0.,  0.]])


PNTMTX_XZ = np.array(
    [[-6.,  6.,  6., -6.],
     [ 0.,  0.,  0.,  0.],
     [-3., -3.,  3.,  3.]])


def test_unit_vecmtx():
    """Test unitization of matrix of direction vectors."""

    unit_vecmtx = mtx_util.unit_vecmtx(mtx_util.vecmtx(PNTMTX_XY))
    chkmtx = np.array(
        [[ 1,  0,  0],
         [ 0,  1,  0],
         [-1,  0,  0],
         [ 0, -1,  0]]).T

    assert np.allclose(unit_vecmtx, chkmtx, atol=1e-10)


def test_normal_vec():
    """Test normal vector associated with a matrix of edge direction vectors."""

    # test xy plane
    nvec = mtx_util.normal_vec(mtx_util.vecmtx(PNTMTX_XY))
    nvec = nvec / np.linalg.norm(nvec)
    chkvec = np.array([0, 0, 1])

    print(nvec)

    assert np.allclose(nvec, chkvec, atol=1e-10)

    # test xz plane
    nvec = mtx_util.normal_vec(mtx_util.vecmtx(PNTMTX_XZ))
    nvec = nvec / np.linalg.norm(nvec)
    chkvec = np.array([0, -1, 0])

    assert np.allclose(nvec, chkvec, atol=1e-10)


def test_ortho_basis_mtx():
    """Test ortho basis vectors."""

    basis_mtx = mtx_util.ortho_basis_mtx(mtx_util.vecmtx(PNTMTX_XZ))
    chk_mtx = np.array(
        [[1,  0,  0],
         [0,  0, -1],
         [0,  1,  0]])

    assert basis_mtx.shape[0] == 3
    assert basis_mtx.shape[1] == 3
    assert np.allclose(basis_mtx, chk_mtx, atol=1e-10)


def test_change_of_basis():
    """Testing change of basis."""

    poly = np.array(
        [[-6.,  6.,  -4., -6.],
        [ 10.,  10.,   10.,  10.],
        [-3., -3.,  15.,  3.]])

    basis_mtx = mtx_util.ortho_basis_mtx(mtx_util.vecmtx(poly))
    inv_basis_mtx = np.linalg.pinv(basis_mtx)

    # change to poly basis
    polyb = np.matmul(inv_basis_mtx, poly)

    assert np.allclose(poly, np.matmul(basis_mtx, polyb), atol=1e-10)

