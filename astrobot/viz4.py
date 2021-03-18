import matplotlib.pyplot as plt
from matplotlib import cm as mpl_cm
from matplotlib import colors as mpl_colors

from shapely import geometry as geomsh

from typing import Optional, List

# torch imports
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform, \
    RasterizationSettings, MeshRenderer, MeshRasterizer, HardPhongShader, \
    PointLights


def get_render_device() -> str:
    # Setup
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    return device

def trimesh_to_polysh(trimesh):
    """Convert mesh to shapely geometries for visualization."""
    return [geomsh.Polygon(tri) for tri in trimesh.triangles]


def simple_cam(pytmeshes: Meshes, dist: float = 80, elev: float = 0, azim: float = 0,
               fov: float = 30, device: Optional[str] = None,
               cam_pt: Optional[List[float]] = None,
               light_pt: Optional[List[float]] = None) -> torch.Tensor:

    """
    Wrapper visualization function to generate simple image from auto-generated camera.

    # TODO: seperate out objecst as args.

    Args:
        pytmeshes: Meshes objects to visualize.

    Returns:
        tensor of image pixels. This can be plotted with:

        .. code-block:: python

            plt.figure(figsize=(3, 3))
            plt.imshow(images[0, ..., :3].cpu().numpy())
            plt.grid(False)

    """

    if device is None:
        device = get_render_device()

    if light_pt is None:
        light_pt = (0, 0, 10)  # 3rd component = z which is front/back
    light_pt = (light_pt, )

    if cam_pt is None:
        cam_pt = (0, 5, 0)  # moves object "up" in image plane by 5.
    cam_pt = (cam_pt, )

    # initialize a camera
    # look_at_view_transform changes location of object relative to camera:
    # y changeS height, z "closeness"
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, at=cam_pt)
    cameras = FoVPerspectiveCameras(
        device=device, R=R, T=T, znear=0.01, fov=fov)
    raster_settings = RasterizationSettings(
        image_size=512, blur_radius=0.0, faces_per_pixel=1)

    # Place a point light in front of the object.
    lights = PointLights(
        device=device, location=light_pt)

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(
            device=device, cameras=cameras, lights=lights)
        )

    # generate image
    images = renderer(pytmeshes)

    return images

