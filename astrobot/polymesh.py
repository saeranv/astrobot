import torch
from functools import lru_cache

import numpy as np
from shapely.geometry import Polygon
import trimesh
Trimesh = trimesh.Trimesh
from typing import Tuple, List, Optional, Union

from . import mtx_util

# torch
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import TexturesVertex


RGB_DICT = {
    'blue': np.array([153, 255, 255]),
    'red': np.array([255, 0, 0]),
    'gold': np.array([255, 215, 0])
    }

DEFAULT_RGB = RGB_DICT['blue'] / 255.0


def DEFAULT_RENDER_DEVICE() -> str:
    # Setup
    if torch.cuda.is_available():
        return 'cuda:0'
    else:
        return 'cpu'


class PolyMesh(object):
    """Polygon and tensor-based mesh representation for deep learning.

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
        self._pytmesh = None

        # private mesh properties
        self._priv_mesh2 = None
        self._priv_mesh3 = None

        # public mesh generation properties
        self._mesh_res = ""  # maximum area of each cell

        # tensor-based mesh

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
    def _mesh2(self):
        """Mesh pntmtx in surface basis (2 x n)."""

        if self._priv_mesh2 is None:
            self._priv_mesh3, self._priv_mesh2 = self._compute_mesh(self._mesh_res)
        return self._priv_mesh2

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
    def pytmesh(self):
        """Get Pytorch3d Mesh object."""

        if self._pytmesh is None:
            raise Exception('pytmesh must be set with `_compute_pytmesh`.')

        return self._pytmesh

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

    def rgb_mtx(self, rgb_arr: Union[List, np.ndarray, Tuple] = DEFAULT_RGB
                ) -> np.ndarray:
        """Make a rgb matrix for PolygonMesh mesh vertices from a rgb color and vertice shape.

        Args:
            rgb_arr: rgb as array i.e (0, 0, 1), values must be normalized between 0 and 1.
                If None, will set as blue.
        Returns:
            rgb matrix shaped by according to mesh vertices.
        """
        # Tuple of mesh vertice shape.
        return np.full(self._mesh3.vertices.shape, rgb_arr)

    def compute_pytmesh(self, texture: TexturesVertex, device: str) -> Meshes:
        """Compute, and set the `pytmesh` property with a pytorch3d Meshes object.

        Note: Mesh coordinates are transformed to camera coordinates, where the z-axis is
        located at the world coordinates y-axis.

        Args:
            texture: TexturesVertex object.
            device: String for render device.

        Returns:
            Meshes object.
        """
        verts = self._verts_to_tensor(self._mesh3.vertices)
        faces = self._faces_to_tensor(self._mesh3.vertices, self._mesh3.faces)
        verts = np.matmul(mtx_util.cam_coord_mtx(), verts.T).T.float()

        self._pytmesh = Meshes(
            verts=[verts.to(device)], faces=[faces.to(device)], textures=texture)

        return self._pytmesh

    @staticmethod
    def _verts_to_tensor(vertices):
        """Convert array of 3d vertice indices to tensor (V, 3)."""

        return torch.tensor(vertices, dtype=torch.float32)

    @staticmethod
    def _faces_to_tensor(vertices, faces):
        """Convert array of face indices to tensor.

        Reference: _format_faces_indices from pytorch3d
        """

        face_idxs = torch.tensor(faces, dtype=torch.int64)

        # check indices
        max_index = vertices.shape[0]

        # check face_indices
        mask = torch.ones(face_idxs.shape[:-1]).bool()  # Keep all faces
        if torch.any(face_idxs[mask] >= max_index) or torch.any(face_idxs[mask] < 0):
            raise Exception("Faces have invalid indices")

        return face_idxs


# TODO: better to disaggregate these two functions into polymesh.
def pytmeshes_textures(rgb_mtx, device: Optional[str] = None) -> TexturesVertex:
    """Make a rgb matrix for mesh vertices from rgb_array.

    Must be transformed to torch tensor to be converted to TextureVertices object.
    Reference: https://github.com/facebookresearch/pytorch3d/issues/51

    Args:
        rgb_mtx: rgb matrix shaped by according to mesh vertices.

    Returns:
        TexturesVertex object.
    """

    if device is None:
        device = DEFAULT_RENDER_DEVICE()

    return TexturesVertex(verts_features=torch.tensor(np.array(rgb_mtx)).to(device))


def pytmeshes(polymeshes, textures: Optional[TexturesVertex] = None,
              device: Optional[str] = None) -> Meshes:
    """List of trimesh objects to pytorch3d Meshes.

    Args:
        polymeshes: List of polymeshes.
        textures: TexturesVertex object. Can construct with `polymesh.mesh_textures`
            function.
        device: Optional string for rendering device.

    Returns:
        Pytorch3d Meshes object.
    """

    if device is None:
        device = DEFAULT_RENDER_DEVICE()

    if textures is None:
        textures = pytmeshes_textures(rgb_mtx=[m.rgb_mtx() for m in polymeshes],
                                      device=device)

    mesh_lst = [pmesh.compute_pytmesh(texture, device)
                for pmesh, texture in zip(polymeshes, textures)]

    if len(mesh_lst) == 1:
        return mesh_lst[0]

    return join_meshes_as_scene(mesh_lst)
