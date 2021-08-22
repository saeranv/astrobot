from typing import List, Union, Tuple, Optional
import os

import numpy as np
import shapely.geometry as geom
from . import mtx_util

import ladybug_geometry.geometry3d as geom3
import ladybug_geometry.geometry2d as geom2
Face3D = geom3.face.Face3D
Point3D = geom3.pointvector.Point3D


def poly_sh(xy_arr):
    """Shapely polygon from list of two arrays of x, y coordinates.

        Args:
            xy_arr: [x_arr, y_arr]
    Example:
        to_poly_sh(to_poly_np(poly_sh))
        to_poly_np(to_poly_sh(poly_np))
    """
    return geom.Polygon([(x, y) for x, y in zip(*xy_arr)])


def poly_np(poly_sh):
    """Two arrays of x, y coordinates from shapely polygon.

    Example:
        to_poly_sh(to_poly_np(poly_sh))
        to_poly_np(to_poly_sh(poly_np))
    """
    return np.array(poly_sh.exterior.xy)


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


def face_to_polysh(face: Face3D) -> geom.Polygon:
    """Convert ladybug Face3D to shapely geom.Polygon."""

    return geom.Polygon(np.array([v.to_array() for v in face.vertices])[:, :2])


def _rect_coords(aspect_ratio, dim):
    """Define rectange coordinates from aspect ratio and long-axis dimension."""

    ew, ns = aspect_ratio * dim, dim
    flr_verts = np.array([
        [-ew, ew, ew, -ew],
        [-ns, -ns, ns, ns],
        [0, 0, 0, 0]])

    return flr_verts / 2.0


def face_from_params(theta: float, area: float = 72, dim: float = 12,
                     aspect_ratio: float = 0.5) -> Face3D:
    """Compute floor vertices from theta, area, dimension, and aspect_ratio."""

    origin = geom3.pointvector.Point3D(0, 0, 0)
    zvec = geom3.pointvector.Vector3D(0, 0, 1)
    plane = geom3.plane.Plane(zvec, origin.duplicate())

    # Make floor
    flr_verts = _rect_coords(aspect_ratio, dim)
    flr_verts = [geom3.pointvector.Point3D(
        *flr_vert) for flr_vert in flr_verts.T]
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
    # zones = [[list(arr) + [0] for arr in poly.to_array()]
    #         for poly in polylst]
    sub_polys = [face.duplicate()]

    return sub_polys


def viz_poly_sh_arr(poly_sh_arr, a=None, scale=1, **kwargs):
    """Plot shapely polygons"""
    if a is None:
        _, a = plt.subplots()
    for poly_sh in poly_sh_arr:
        if poly_sh.exterior:
            xx, yy = poly_sh.exterior.xy
            xx, yy = np.array(xx), np.array(yy)
            a.plot(xx * scale, yy * scale, **kwargs)
    a.axis('equal')
    return a


def to_poly_sh(xy_arr):
    """Shapely polygon from list of two arrays of x, y coordinates.

        Args:
            xy_arr: [x_arr, y_arr]
    Example:
        to_poly_sh(to_poly_np(poly_sh))
        to_poly_np(to_poly_sh(poly_np))
    """
    return geom.Polygon([(x, y) for x, y in zip(*xy_arr)])


def to_poly_np(poly_sh):
    """Two arrays of x, y coordinates from shapely polygon.

    Example:
        to_poly_sh(to_poly_np(poly_sh))
        to_poly_np(to_poly_sh(poly_np))
    """
    return np.array(poly_sh.exterior.xy)
