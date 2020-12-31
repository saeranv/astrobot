import numpy as np
import astrobot.geom_util

from dragonfly import room2d, story, building, model, windowparameter
from honeybee_energy import lib, constructionset, material, construction

# List of program types available in honeybee_energy
PROGRAM_TYPES = lib.programtypes.PROGRAM_TYPES


def program_space_types(program_type="Office"):
    """Filters program space types by program string."""
    def _chk_type(pt, st):
        if '::' in pt or '::' not in st:
            return pt in st
        else:
            return pt in st.split('::')[1]

    pt = program_type
    return [st for st in PROGRAM_TYPES if _chk_type(pt, st)]


def program_space_types_blend(space_types, weights, space_type_name='blended'):
    space_types = [lib.programtypes.program_type_by_identifier(st) for st in space_types]
    return lib.programtypes.ProgramType.average(space_type_name, space_types, weights)


def set_room2d_program_space_type(room, program_type):
    """Assign program to space-type."""
    room.properties.energy.program_type = program_type


def face3d_to_room2d(face3d, ht=3.5, id='null'):
    """Make ladybug Room2D from face3d"""
    room = room2d.Room2D(id, face3d, floor_to_ceiling_height=ht)
    room.properties.energy.add_default_ideal_air()

    return room


def face3ds_to_room2ds(face3ds, face3d_idxs, mod_idx):
    """Convert a matrix of space indices and face3ds to room2ds for a single model."""
    room2ds = [0] * len(face3d_idxs)
    for i in range(len(face3d_idxs)):
        spc_geom, spc_idx = face3ds[i], face3d_idxs[i]
        spc_id = 'mod_{}_spc_{}'.format(mod_idx, spc_idx)
        room2ds[i] = face3d_to_room2d(spc_geom, 3.5, spc_id)

    return room2ds


def set_room2d_wwr(room, wwr_theta, wwr):
    """Set wwr to room2d based on reference angle of wall."""

    seg_thetas = room.segment_orientations()
    win_params = list(room.window_parameters)
    bcs = room.boundary_conditions

    for i in range(len(seg_thetas)):
        is_out_wwr_theta = \
            (np.abs(seg_thetas[i] - wwr_theta) < 1e-10) and (bcs[i].name == 'Outdoors')
        if is_out_wwr_theta:
            win_params[i] = windowparameter.SimpleWindowRatio(wwr)

    room.window_parameters = win_params


def set_building_wwr(building, wwr, wwr_theta):
    """Assign window ratio to building."""

    for story in building:
        for room in story.room_2ds:
            set_room2d_wwr(room, wwr_theta, wwr)


def construction_set(u_value=2.0):
    """Construction set model.
    TODO: make single pane window construction:
        - build up assembly/gas gaps/low-e + Energyplus has separate frame/divider objects
        - not implemented at this point in honeybee-energy

    TODO: Apply construction set for office program
        - https://github.com/ladybug-tools/honeybee-energy-standards/blob/master/honeybee_energy_standards/data/constructionsets/2013_data.json
        -"2013::ClimateZone1::SteelFramed"
    """
    # get generic construction set
    constr_set = lib.constructionsets.generic_construction_set.duplicate()
    from honeybee_energy.material.glazing import EnergyWindowMaterialSimpleGlazSys
    from honeybee_energy.construction.window import WindowConstruction

    # assign window
    glass = EnergyWindowMaterialSimpleGlazSys(
        'window_simple_glazing_system', u_factor=u_value, shgc=0.35, vt=0.7)
    window = WindowConstruction('Window Construction', [glass])
    constr_set.aperture_set.window_construction = window

    return constr_set

def set_model_wwr(model, wwrs, thetas):

    for bld in model.buildings:
        set_building_wwr(bld, wwrs, thetas)

def set_model_construction_set(model):

    for bld in model.buildings:
        bld.properties.energy.construction_set = construction_set()


def room2ds_to_model(room2ds, id='null'):
    """Make simple model from list of Room2Ds.
    # https://github.com/ladybug-tools/honeybee-grasshopper-core/blob/master/honeybee_grasshopper_core/src/HB%20Apertures%20by%20Ratio.py#L135-L139

    TODO: solve adjacencies.
    # room.is_groundcontact, is_top_in_contact (in Story object)
    # story.multiplier = 5, then separate into top_bottom_floors()
    # all_stories() if you need all of them
    """

    # room2ds to story to building to model
    _story = story.Story(id + "_story", room2ds)
    _story.solve_room_2d_adjacency(0.01)
    _building = building.Building(id + '_building', [_story])

    return model.Model(id + '_model', [_building], tolerance=0.01)
