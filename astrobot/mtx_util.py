import numpy as np

DEFAULT_TOLERANCE = 1e-10


"""
Notes on vec and pnt distinction:

vec refers to vectors, typically referring to surface edge direction vectors.
pnt refers to points, typically referring to surface vertices.

With respect to transformations, geometric transformations are 4 x 4 matrices, with the
last coordinate set as a 0 for vectors, and 1 for points. This is to allow the
translation of points (since the 4th column of a 4x4 matrix is its translation vector),
but not vectors. Note that both vectors and points are affected by the rotation, scaling
and other transformations that are not limited to the last column.

Notes on numpy matrices:

Numpy matrices are row major, but for legibility pnt and vec use row for x, y, z
coordinates. This is why two transposes are needed during matrix transformations.

.. code-block:: python

    mtx = np.array(
        [[-6.,  6.,  6., -6.],
        [ 0.,  0.,  0.,  0.],
        [-3., -3.,  3.,  3.]])

    >> m[:, 0]
    [-6, 0, -3]

    >> m[:, 0:1]
    [[-6.],
    [ 0.],
    [-3.]])

"""


def vecmtx(pntmtx: np.ndarray) -> np.ndarray:
    """Compute a matrix of non-unit direction vectors associated with pntmtx.

    Can be considered the directed edges of from an array of vertices.

    Args:
        pntmtx: Array of surface vertices.

    Returns:
        Array of vc vectors:

        .. code-block:: python

            [vertice1 - vertice0,  # vec1
             vertice2 - vertice1,  # vec2
             ...,
             verticen - verticen-1]  # vecn
    """

    # make loop
    _pntmtx = np.concatenate((pntmtx, pntmtx[:, 0:1]), axis=1)

    # n x 2 x 3 matrix, where n: |pntmtx|
    vecmtx = [(_pntmtx[:, i], _pntmtx[:, i + 1])
              for i in range(_pntmtx.shape[1] - 1)]
    vecmtx = np.array(vecmtx)

    # subtract vertex_n from vertex_n-1, and transpose
    return (vecmtx[:, 1, :] - vecmtx[:, 0, :]).T


def unit_vecmtx(vecmtx: np.ndarray) -> np.ndarray:
    """Unitize vc vectors in vecmtx.

    This is a somewhat contrived way to use matrix transformation to normalize an array
    of vecs. But a good test of matrix intuition. The ith col vectors in the divmtx
    multiplies by the jth column in the vecmtx s.t. each dir vector is divided by its
    magnitude.
    """
    # create a n x n matrix to divide the vc vectors with
    divmtx = np.diag(1.0 / np.linalg.norm(vecmtx, axis=0))

    # (3 x n) * (n x n)
    return np.matmul(vecmtx, divmtx)


def corr(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Correlation (between 0 and 1) for two vectors.

    Args:
        vec1: First vector as numpy array. Does not need to be unitized.
        vec2: Second vector as numpy array. Does not need to be unitized.

    Return:
        Float between 0 and 1 representing correlation between vectors.
    """
    # interpretation from G. Strang: dot(V, V) is better thought of as a linear
    # transformation of two vectors to 1 dimensional space: matmul(V^T,  V).
    return np.matmul(vec1 / np.linalg.norm(vec2).T, vec2 / np.linalg.norm(vec2))


def is_parallel(vec1: np.ndarray, vec2: np.ndarray, tol: float = DEFAULT_TOLERANCE
                ) -> bool:
    """Boolean classifying if two vectors are parrallel."""
    return np.abs(1.0 - corr(vec1, vec2)) < tol


def ortho_basis_mtx(vecmtx: np.ndarray, calc_only_normal: bool = False) -> np.ndarray:
    """Compute the orthogonal change of basis matrix for the surface defined by vecmtx.

    It consists of an array of column, unit vectors of the x, y and normal vector of the
    vecmtx. This matrix transforms point coordinates in the basis space back to the
    standard basis space. Use the inverse of this matrix (i.e.
    `np.linalg.pinv(ortho_basis_mtx)`) to get a transformation matrix to transform from
    standard basis to the basis of the surface with:

    Args:
        vecmtx: Array of vectors.
        calc_only_normal: Optional argument to compute just the normal vector.
            Default: False.

    Returns:
        Array of x-axis vector, y-axis vector and normal vector, or
            just the normal vector if calc_only_normal is True. This
            constitutes the change of basis matrix.
    """

    def _is_parallel(d): return np.abs(1.0 - d) < DEFAULT_TOLERANCE

    vec_idx = 0
    d = corr(vecmtx[:, vec_idx], vecmtx[:, vec_idx + 1])

    # find non-collinear vecs
    while _is_parallel(d) and vec_idx < vecmtx.shape[0]:
        vec_idx += 1
        d = corr(vecmtx[:, vec_idx], vecmtx[:, vec_idx + 1])

    # Note that the ccw order of the polygon edges results in a
    # +z-axis normal for a xy-plane, and a -y-axis normal for a
    # xz-plane. This corresponds with this module's assumptions
    # of outward facing normals.
    nvec = np.cross(vecmtx[:, vec_idx], vecmtx[:, vec_idx + 1])

    if calc_only_normal:
        return nvec

    # Otherwise return the yvec orthogonal to the xvec by first checking
    # if the correlation is 0 (perpendicular), else computing the cross
    # between the xvec and normal.
    xvec = vecmtx[:, vec_idx]
    if np.abs(d) < DEFAULT_TOLERANCE:
        yvec = vecmtx[:, vec_idx + 1]
    else:
        yvec = np.cross(nvec, xvec)

    # transpose to ensure basis vectors are swapped to columns
    return unit_vecmtx(np.array([xvec, yvec, nvec]).T)


def normal_vec(vecmtx: np.ndarray) -> np.ndarray:
    """Compute the (non-unit) normal vector associated with a (non-unit) vecmtx.

    Normal is computed by traversing vecs until it finds non-collinear vecs.
    The normal vector is perpendicular to the plane between the two non-collinear
    vectors.

    Args:
        vecmtx: Array of vectors.

    Returns:
        Normal vector associated with vecmtx.
    """
    return ortho_basis_mtx(vecmtx, calc_only_normal=True)


def cam_coord_mtx() -> np.ndarray:
    """matrix to transform world coordinates to camera coordinates.

    For camera coordinate systems the x-axis stays the same, but the y-axis rotates to
    the z-axis, and the z-axis is rotated to the negative y-axis. Thus the negative
    z-axis is the "front" of an object.
    """

    return np.array(
        [[1., 0., 0.],
         [0., 0., -1.],
         [0., 1., 0.]]
    )


def contiguous_ones_idx(bin_arr: np.ndarray) -> np.ndarray:
    """Find indices of contiguous (repeated) ones in binary array.

    Given binary array (0s, 1s), returns (start, end) indices of
    repeated blocks of ones:

    bin_arr = np.array([1, 0, 0, 1, 0, 0, 1, 1])
    contiguous_ones_idx(bin_arr) -> [[0, 1] [3, 4] [6, 8]]
    """
    bin_arr = (bin_arr / bin_arr.max()).astype(np.uint8)
    bin_arr = np.concatenate(([0], bin_arr, [0]))
    diff = np.abs(np.diff(bin_arr))
    zero_idx = np.where(diff == 1)[0].reshape(-1, 2)
    return zero_idx
