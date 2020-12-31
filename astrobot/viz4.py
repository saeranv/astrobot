import matplotlib.pyplot as plt
from matplotlib import cm as mpl_cm
from matplotlib import colors as mpl_colors

from shapely import geometry as geomsh


def mesh_geoms(mesh):
    """Convert mesh to shapely geometries for visualization."""
    return [geomsh.Polygon(tri) for tri in mesh.triangles]
