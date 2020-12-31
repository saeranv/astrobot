import numpy as np
from shapely import geometry
import trimesh

import ladybug_geometry.geometry3d as geom3
import ladybug_geometry.geometry2d as geom2


# TODO: transpose pntmtx
def rect_coords(aspect_ratio, dim):
    """Define rectange coordinates from aspect ratio and long-axis dimension."""
    ns, ew = aspect_ratio * dim, dim
    flr_verts = np.array([
        [-ew, -ns,  0],
        [ ew, -ns,  0],
        [ ew,  ns,  0],
        [-ew,  ns,  0]])
    return flr_verts / 2.0


# TODO: transpose pntmtx
def face_from_params(theta, area=72, dim=12, aspect_ratio=0.5):
    """Compute floor vertices from theta, area, dimension, and aspect_ratio."""
    origin = geom3.pointvector.Point3D(0, 0, 0)
    zvec = geom3.pointvector.Vector3D(0, 0, 1)
    plane = geom3.plane.Plane(zvec, origin.duplicate())

    # Make floor
    flr_verts = rect_coords(aspect_ratio, dim)
    flr_verts = [geom3.pointvector.Point3D(*flr_vert) for flr_vert in flr_verts]
    flr = geom3.face.Face3D(flr_verts, plane).rotate(zvec, theta, origin)

    # Transformations
    scale_fac = np.sqrt(area / flr.area)
    if np.abs(scale_fac - 1.0) > 1e-10:
        flr = flr.scale(scale_fac, origin)

    return flr


# TODO: transpose pntmtx
def polyskel_faces(faces):
    """Split face into polyskeleton.

    Returns:
        Nested list of each model and spaces.
    """
    sub_polys = [0] * len(faces)
    for i in range(len(faces)):
        #core, perimlst, _ = polyskel.sub_polygons(poly2d, 5)
        #polylst = perimlst + [core]
        #zones = [[list(arr) + [0] for arr in poly.to_array()]
        #         for poly in polylst]
        sub_polys[i] = [faces[i].duplicate()]

    return sub_polys


# TODO: transpose pntmtx
def split_mod_to_polyskel(mod_idxs, mod_geoms):
    """Split faces into polyskel spaces.

    Returns:
        Parent index and sub-polygons.
    """
    spc_mtx = polyskel_faces(mod_geoms)
    spc_vec = [_spc for _spc_mtx in spc_mtx for _spc in _spc_mtx]
    spc_mod_idxs = np.zeros(len(spc_vec)).astype(int)

    for i in range(len(spc_mtx)):
        mod_idx, spc_num = mod_idxs[i], len(spc_mtx[i])
        spc_mod_idxs[i:i + spc_num] = [mod_idx for i in range(spc_num)]

    return spc_vec, spc_mod_idxs


# TODO: transpose pntmtx
def triangulate(vertmtx, args=[]):
    """Triangulate a polygon vertmtx, and return as trimesh object.

    Args:
        vertmtx: Numpy array of polygon vertices.
        args: List of arguments as string for triangulate library. Examples from the
            triangle documentation include: 'q0.01' or 'a0.1'.

    Returns:
        A trimesh Trimesh object.

    Note:
        [1] https://rufat.be/triangle/examples.html.
    """
    verts, faces = trimesh.creation.triangulate_polygon(vertmtx, triangle_args=args)
    return trimesh.Trimesh(vertices=verts, faces=faces)
