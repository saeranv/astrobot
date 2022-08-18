import matplotlib.pyplot as plt
from matplotlib import cm as mpl_cm
from matplotlib import colors as mpl_colors

from shapely import geometry as geomsh

from typing import Optional, List

import pandas as pd


class Viz4(object):

    ZONE_LOAD_COLOR_DICT = {
        "cool": "#4542f5",
        "heat": '#700000',
        "shw": "#FF9800",
        "light_electric": "#FFEB3B",
        "equip_electric": "#CDDC39",
        "fan": "#607D8B",
        "pump": "#B0BEC5",
        # Missing
        "occ_heat": '#F542b3',
        "window_transmit_solar": "#F44336",
        "infil_heat": '#706700',
        "mech_vent": "#44FF00",
        "glaze_cond": "#42F5E3",
        "opaque_cond": "#2196F3",
        "envelope": "#A200FF",
        "misc": "#A3A3A3"
    }

    LOAD_PRINT_NAME_DICT = {
        "occ_heat": "Occupant Heat",
        "light_electric": "Light Electric",
        "equip_electric": "Equipment Electric",
        "window_transmit_solar": "Window Transmit Solar",
        'envelope': "Envelope",
        'mech_vent': "Mechanical Ventilation",
        "infil_heat": "Infiltration Heat",
        "heat": "Mechanical Heating",
        "cool": "Mechanical Cooling"
    }

    # # torch imports
    # import torch
    # from pytorch3d.structures import Meshes
    # from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform, \
    #     RasterizationSettings, MeshRenderer, MeshRasterizer, HardPhongShader, \
    #     PointLights

    # def get_render_device() -> str:
    #     # Setup
    #     if torch.cuda.is_available():
    #         device = 'cuda:0'
    #     else:
    #         device = 'cpu'

    #     return device

    # def trimesh_to_polysh(trimesh):
    #     """Convert mesh to shapely geometries for visualization."""
    #     return [geomsh.Polygon(tri) for tri in trimesh.triangles]

    # def simple_cam(pytmeshes: Meshes, dist: float = 80, elev: float = 0, azim: float = 0,
    #                fov: float = 30, device: Optional[str] = None,
    #                cam_pt: Optional[List[float]] = None,
    #                light_pt: Optional[List[float]] = None) -> torch.Tensor:

    #     """
    #     Wrapper visualization function to generate simple image from auto-generated camera.

    #     # TODO: seperate out objecst as args.

    #     Args:
    #         pytmeshes: Meshes objects to visualize.

    #     Returns:
    #         tensor of image pixels. This can be plotted with:

    #         .. code-block:: python

    #             plt.figure(figsize=(3, 3))
    #             plt.imshow(images[0, ..., :3].cpu().numpy())
    #             plt.grid(False)

    #     """

    #     if device is None:
    #         device = get_render_device()

    #     if light_pt is None:
    #         light_pt = (0, 0, 10)  # 3rd component = z which is front/back
    #     light_pt = (light_pt, )

    #     if cam_pt is None:
    #         cam_pt = (0, 5, 0)  # moves object "up" in image plane by 5.
    #     cam_pt = (cam_pt, )

    #     # initialize a camera
    #     # look_at_view_transform changes location of object relative to camera:
    #     # y changeS height, z "closeness"
    #     R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, at=cam_pt)
    #     cameras = FoVPerspectiveCameras(
    #         device=device, R=R, T=T, znear=0.01, fov=fov)
    #     raster_settings = RasterizationSettings(
    #         image_size=512, blur_radius=0.0, faces_per_pixel=1)

    #     # Place a point light in front of the object.
    #     lights = PointLights(
    #         device=device, location=light_pt)

    #     renderer = MeshRenderer(
    #         rasterizer=MeshRasterizer(
    #             cameras=cameras, raster_settings=raster_settings),
    #         shader=HardPhongShader(
    #             device=device, cameras=cameras, lights=lights)
    #         )

    #     # generate image
    #     images = renderer(pytmeshes)

    #     return images

    @staticmethod
    def ezplt_df(df, labels, kind='bar', stacked=True, ax=None, color=None, showlegend=True, legendcol=1):
        if ax is None:
            fig, ax = plt.subplots(1, 1)

        _ax = ax
        _ax.set_xlabel(labels[0])
        _ax.set_ylabel(labels[1])
        _ax = df.plot(
            kind=kind,
            stacked=stacked,
            ax=_ax,
            color=color,
            legend=showlegend,
            # alpha=0.5,
            lw=0.5,
            edgecolor='black'
        )
        _ax.grid()
        _ax.set_axisbelow(True)
        if showlegend:
            _ax.legend(loc='upper center', bbox_to_anchor=(
                0.5, -0.25), ncol=legendcol)

        return _ax

    @classmethod
    def stacked_eui_plot(
        cls,
        df_eui,
        max_eui,
        min_eui=0,
        ax=None,
        xlabel='CBECS Buildings ID',
        ylabel='Energy Use Intensity (kBtu/sf)',
        showlegend=True,
        legendcol=1
    ):
        """
        Consistent bar widths, with dynamically adjusted size.
        :param df_eui: dataframe with rows=buildings, cols=eui_loads
        :param max_eui: maximum eui that sets ylim
        :param xlabel: label for graph x axis, default Buildings
        :param ylabel: lalbel for graph y axis, default EUI
        :return ax: ax
        """
        num_bars = df_eui.values.shape[0]
        axlabels = [xlabel, ylabel]
        colorlst = [cls.ZONE_LOAD_COLOR_DICT[eui_col]
                    for eui_col in df_eui.columns]

        df_eui = df_eui.rename(mapper=cls.LOAD_PRINT_NAME_DICT, axis=1)

        if ax is None:
            fig, ax = plt.subplots(1, 1, False, False, figsize=(num_bars, 5))

        ax = Viz4.ezplt_df(
            df_eui,
            axlabels,
            kind='bar',
            stacked=True,
            ax=ax,
            color=colorlst,
            showlegend=showlegend,
            legendcol=legendcol
        )
        # (self, df, labels, kind='bar', stacked=True, ax=None, color=None, showlegend=True, legendcol=1):
        ax.set_ylim(min_eui, max_eui)
        ax.set_xlim(-0.5, num_bars - 0.5)

        return ax
