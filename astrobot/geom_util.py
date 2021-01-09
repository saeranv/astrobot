import os

import numpy as np
from shapely.geometry import Polygon
from . import mtx_util

import ladybug_geometry.geometry3d as geom3
import ladybug_geometry.geometry2d as geom2
Face3D = geom3.face.Face3D
Point3D = geom3.pointvector.Point3D

from typing import List, Union, Tuple, Optional
from trimesh import Trimesh
from trimesh.exchange.obj import export_obj


# torch
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import TexturesVertex


RGB_DICT = {
    'blue': np.array([153, 255, 255]),
    'red': np.array([255, 0, 0]),
    'gold': np.array([255, 215, 0])
    }


def to_lb_theta(theta: float) -> float:
    """Converts orientation in radians to ladybug's orientation conventions.

    Ladybug orientation is in clockwise degrees.

    Args:
        theta: theta in ccw radians.

    Returns:
        theta in cw degrees.
    """
    return (360.0 - (theta * 180.0 / np.pi)) % 360.0


def verts_to_pntmtx(vertices: List[Point3D]) -> np.ndarray:
    """Convert list of ladybug Point3Ds to pntmtx."""
    return np.array([vertex.to_array() for vertex in vertices]).T


def face_to_polysh(face: Face3D) -> Polygon:
    """Convert ladybug Face3D to shapely Polygon."""

    return Polygon(np.array([v.to_array() for v in face.vertices])[:, :2])


def _rect_coords(aspect_ratio, dim):
    """Define rectange coordinates from aspect ratio and long-axis dimension."""

    ew, ns = aspect_ratio * dim, dim
    flr_verts = np.array([
        [-ew,  ew, ew, -ew],
        [-ns, -ns, ns,  ns],
        [  0,   0,  0,   0]])

    return flr_verts / 2.0


def face_from_params(theta: float, area: float = 72, dim: float = 12,
                     aspect_ratio: float = 0.5) -> Face3D:
    """Compute floor vertices from theta, area, dimension, and aspect_ratio."""

    origin = geom3.pointvector.Point3D(0, 0, 0)
    zvec = geom3.pointvector.Vector3D(0, 0, 1)
    plane = geom3.plane.Plane(zvec, origin.duplicate())

    # Make floor
    flr_verts = _rect_coords(aspect_ratio, dim)
    flr_verts = [geom3.pointvector.Point3D(*flr_vert) for flr_vert in flr_verts.T]
    flr = Face3D(flr_verts, plane).rotate(zvec, theta, origin)

    # Transformations
    scale_fac = np.sqrt(area / flr.area)
    if np.abs(scale_fac - 1.0) > 1e-10:
        flr = flr.scale(scale_fac, origin)

    return flr


def polyskel_from_face(face: Face3D) -> List[Face3D]:
    """Split face into polyskeleton.

    Returns:
        List of polyskel polygons as faces.
    """

    #core, perimlst, _ = polyskel.sub_polygons(poly2d, 5)
    #polylst = perimlst + [core]
    #zones = [[list(arr) + [0] for arr in poly.to_array()]
    #         for poly in polylst]
    sub_polys = [face.duplicate()]

    return sub_polys


def get_render_device() -> str:
    # Setup
    if torch.cuda.is_available():
        #device = torch.device("cuda:0")
        #torch.cuda.set_device(device)
        device = 'cuda:0'
    else:
        #device = torch.device("cpu")
        device = 'cpu'

    return device


def rgbs_to_textures(verts_shapes: List[tuple], rgb_arrs:
                     Union[List, np.ndarray, None] = None, device: str = 'cpu'
                     ) -> TexturesVertex:
    """Make a rgb matrix for mesh vertices from rgb_array.

    Must be transformed to torch tensor to be converted to TextureVertices object.
    Reference: https://github.com/facebookresearch/pytorch3d/issues/51

    Args:
        verts_shape: List of mesh vertice shape tuples.
        rgb_arrs: List of rgb arrays i.e [[0, 0, 1]]. Rgb value must be normalized
            between 0 and 1.

    Returns:
        TexturesVertex object.
    """
    msh_num = len(verts_shapes)

    if rgb_arrs is None:
        _default_rgb = RGB_DICT['blue'] / 255.0
        rgb_arrs = [_default_rgb for _ in range(msh_num)]

    textures = [0] * msh_num
    for i, (verts_shape, rgb_arr) in enumerate(zip(verts_shapes, rgb_arrs)):
        textures[i] = np.full(verts_shape, rgb_arr)

    return TexturesVertex(verts_features=torch.tensor(np.array(textures)).to(device))


def trimesh_to_pytmesh(trimesh: Trimesh, device: str = 'cpu'
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute tensors for mesh vertices and face indices from Trimesh.

    Ref: https://github.com/facebookresearch/pytorch3d/blob/\
        1f9cf91e1b59c1b140efaf9f384e6248c4a125d8/pytorch3d/io/obj_io.py#L26
    """
    verts = torch.tensor(trimesh.vertices, dtype=torch.float32, device=device)  # (V, 3)

    # ref: _format_faces_indices from pytorch3d
    face_idxs = torch.tensor(trimesh.faces, dtype=torch.int64, device=device)

    # check indices
    max_index = verts.shape[0]
    # check face_indices
    mask = torch.ones(face_idxs.shape[:-1]).bool()  # Keep all faces
    if torch.any(face_idxs[mask] >= max_index) or torch.any(face_idxs[mask] < 0):
        raise Exception("Faces have invalid indices")

    return verts, face_idxs


def trimeshes_to_pytmeshes(trimeshs: List[Trimesh],
                           textures: Optional[TexturesVertex] = None,
                           device: Optional[str] = None) -> Meshes:
    """List of trimesh objects to pytorch3d Meshes.

    Args:
        verts_rgb_tensor: Torch tensor (m x n x 3) of rgb values for each mesh, and
            mesh vertices: torch.tensor([[r, g, b], ...]. Each rgb value is
            normalized between 0 and 1.

    Returns:
        Pytorch3d Meshes object.
    """
    mesh_lst = []

    if device is None:
        device = get_render_device()

    if textures is None:
        textures = rgbs_to_textures([m.vertices.shape for m in trimeshs], None, device)

    for i, (trimesh, texture) in enumerate(zip(trimeshs, textures)):

        verts, face_idxs = trimesh_to_pytmesh(trimesh, device)

        # Transform mesh coordinates to camera coordinates (with z-axis at world
        # coordinates y-axis).
        verts = np.matmul(mtx_util.cam_coord_mtx(), verts.T).T.float()

        mesh = Meshes(
            verts=[verts.to(device)],
            faces=[face_idxs.to(device)],
            textures=texture
        )
        mesh_lst.append(mesh)

    if len(mesh_lst) == 1:
        return mesh_lst[0]

    return join_meshes_as_scene(mesh_lst)


def trimesh_to_pytmeshs_with_obj(trimesh: Trimesh, device: str = 'cpu'
                                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate pytorch3d Meshes from a single trimesh by exporting and loading obj.

    Used solely for testing.

    Args:
        trimesh: Trimesh object.

    Returns:
        vertices, and face indices as tuple of torch tensors.
    """
    obj = export_obj(trimesh)
    mesh_fpath = os.path.abspath(os.path.join(os.getcwd(), 'mesh_0.obj'))
    with open(mesh_fpath, 'w') as fp:
        fp.writelines(obj)
    verts, faces_idx, _ = load_obj(mesh_fpath, device=device)
    faces = faces_idx.verts_idx
    os.remove(mesh_fpath)

    return verts, faces

