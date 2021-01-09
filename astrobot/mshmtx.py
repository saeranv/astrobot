from functools import lru_cache

import numpy as np
from shapely.geometry import Polygon
import trimesh
Trimesh = trimesh.Trimesh
from typing import Tuple

import astrobot.mtx_util as mtx_util
import astrobot.geom_util as geom_util


class MshMtx(object):
    """Mtx stores polygon geometric properties from matrix computations of an array of points.

    Args:
        pntmtx: Array of points.
    """
    def __init__(self, pntmtx: np.ndarray):

        self.pntmtx = pntmtx

        # init empty properties to be dynamically computed
        self._vecmtx = None
        self._basis_mtx = None
        self._inv_basis_mtx = None
        self._pntmtx2 = None
        self._polysh = None
        self._priv_mesh2 = None
        self._priv_mesh3 = None

        #
        self._mesh_res = ""  # maximum area of each cell

    def __repr__(self):
        return 'MshMtx:\n' + str(self.pntmtx.round(2))

    @property
    def vecmtx(self):
        """Array of edge vectors of polygon."""

        if self._vecmtx is None:
            self._vecmtx = mtx_util.vecmtx(self.pntmtx)
        return self._vecmtx

    @property
    def basis_mtx(self):
        """Array of orthogonal basis vectors: x, y, and normal."""

        if self._basis_mtx is None:
            self._basis_mtx = mtx_util.ortho_basis_mtx(self.vecmtx)
        return self._basis_mtx

    @property
    def inv_basis_mtx(self):
        """Inverse basis matrix, used for change of basis transformations."""

        if self._inv_basis_mtx is None:
            self._inv_basis_mtx = np.linalg.pinv(self.basis_mtx)
        return self._inv_basis_mtx

    @property
    def pntmtx2(self):
        """Array of 2d points (3 x n where 3rd row is 0) within basis space of polygon plane.

        The pntmtx is transformed from the standard basis to a basis of any arbitrarily
        oriented surface by mulitiplying the points by an inverse basis matrix.
        """
        if self._pntmtx2 is None:
            self._pntmtx2 = np.matmul(self.inv_basis_mtx, self.pntmtx)
        return self._pntmtx2

    @property
    def polysh(self):
        """Polygon as shapely Polygon object.

        Shapely Polygons are 2d, so polysh is defined by pntmtx2.
        """

        if self._polysh is None:
            self._polysh = Polygon(self.pntmtx2.T)
        return self._polysh

    @property
    def _mesh3(self):
        """Triangular mesh of polygon, as a Trimesh object.

        Note the mesh computation function is memoized, so triangle
        arguments are cached.
        """

        if self._priv_mesh3 is None:
            self._priv_mesh3, self._priv_mesh2 = self._compute_mesh(self._mesh_res)
        return self._priv_mesh3

    @property
    def _mesh2(self):
        """Mesh pntmtx in surface basis (2 x n)."""

        if self._priv_mesh2 is None:
            self._priv_mesh3, self._priv_mesh2 = self._compute_mesh(self._mesh_res)
        return self._priv_mesh2

    @lru_cache(maxsize=128)
    def _compute_mesh(self, mesh_args: str = "") -> Tuple[Trimesh]:
        """Triangulate a polygon vertmtx, and return as trimesh object.

        Memoized by string argument.

        Args:
            mesh_args: List of arguments as string for triangulate library. Examples from
                the triangle documentation include: 'q0.01' or 'a0.1'.

        Returns:
            A trimesh Trimesh object for mesh in 2d and 3d.

        Note:
            [1] https://rufat.be/triangle/examples.html.
        """
        process = True

        # make mesh2
        mesh_pntmtx2, mesh_faces = \
            trimesh.creation.triangulate_polygon(self.polysh, triangle_args=mesh_args)
        mesh2 = Trimesh(vertices=mesh_pntmtx2, faces=mesh_faces, process=process)
        mesh_pntmtx2 = mesh_pntmtx2.T

        # make mesh3
        # change from srf basis back to standard basis, and add back
        # third dimension from cached attributes.
        mesh_pntmtx3 = np.concatenate(
            (mesh_pntmtx2, (self.pntmtx2[2, :].T,)), axis=0)
        mesh_pntmtx3 = np.matmul(self.basis_mtx, mesh_pntmtx3)
        mesh3 = Trimesh(
            vertices=mesh_pntmtx3.T, faces=mesh_faces, process=process)

        return mesh3, mesh2
