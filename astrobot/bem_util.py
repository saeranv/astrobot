import numpy as np

from . import geom_util

from dragonfly import room2d, story, building, model, windowparameter

from honeybee import boundarycondition as bc
from honeybee import facetype
from honeybee_energy import lib


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
    space_types = [lib.programtypes.program_type_by_identifier(
        st) for st in space_types]
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


def set_room2d_wwr(room, wwr, wwr_theta=None):
    """Assign window ratio to room2ds based on reference angle of wall."""

    win_params = list(room.window_parameters)
    bcs = room.boundary_conditions
    seg_thetas = room.segment_orientations()

    wwr_theta = geom_util.to_lb_theta(wwr_theta)
    for i, seg_theta in enumerate(seg_thetas):
        if bcs[i].name != 'Outdoors':
            continue

        if wwr_theta:
            is_out_wwr_theta = (np.abs(seg_theta - wwr_theta) < 1e-10)
            if not is_out_wwr_theta:
                continue

        win_params[i] = windowparameter.SimpleWindowRatio(wwr)

    room.window_parameters = win_params


def set_building_wwr(building, wwr, wwr_theta):
    """Assign window ratio to building consisting of multiple stories."""

    for building_story in building:
        for room in building_story.room_2ds:
            set_room2d_wwr(room, wwr_theta, wwr)


def set_model_wwr(model, wwrs, thetas):
    """Assign window ratio to model consisting of multiple buildings."""
    [set_building_wwr(bld, wwrs, thetas) for bld in model.buildings]


def construction_set(u_value=2.0):
    """Construction set model.
    TODO: make single pane window construction:
        - build up assembly/gas gaps/low-e + Energyplus has separate frame/divider
          objects
        - not implemented at this point in honeybee-energy

    TODO: Apply construction set for office program
        - https://github.com/ladybug-tools/honeybee-energy-standards/blob/master/\
            honeybee_energy_standards/data/constructionsets/2013_data.json
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


def set_model_construction_set(model):

    for bld in model.buildings:
        bld.properties.energy.construction_set = construction_set()


def room2ds_to_model(room2ds, id='null'):
    """Make simple model from list of Room2Ds.
    # https://github.com/ladybug-tools/honeybee-grasshopper-core/blob/master/\
    # honeybee_grasshopper_core/src/HB%20Apertures%20by%20Ratio.py#L135-L139

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


def spcs_from_mods(mods, mod_idxs):
    """Retrieve spaces and associated model indices from list of models.

    Args:
        mods: List of Honeybee models.
        mod_idxs: list of model indices.

    Returns:
        Tuple of:
        * List of model indices for all spaces in model.
        * List of all spaces in model.
    """

    spc_mod_idx, spcs = [], []

    for mod_idx, mod in zip(mod_idxs, mods):
        _spcs = mod.rooms
        spc_mod_idx.extend([mod_idx] * len(_spcs))
        spcs.extend(_spcs)

    return np.array(spc_mod_idx, dtype=int), spcs


def srfs_from_spcs(spcs, spc_idxs, spc_mod_idxs):
    """Retrieve surfaces and associated space model indices from list of spaces.

    Args:
        spcs: List of spaces.
        spc_idxs: List of space indices.
        spc_mod_idxs: List of model indices for each space.

    Returns:
        Tuple of:
        * List of mod indices for surfaces.
        * List of spc indices for surfaces.
        * List of all surfaces.
    """

    srf_mod_idx, srf_spc_idx, srfs = [], [], []

    for spc_mod_idx, spc_idx, spc in zip(spc_mod_idxs, spc_idxs, spcs):

        _srfs = spc.faces
        _len_srfs = len(_srfs)
        srf_mod_idx.extend([spc_mod_idx] * _len_srfs)
        srf_spc_idx.extend([spc_idx] * _len_srfs)
        srfs.extend(_srfs)

    return (np.array(srf_mod_idx, dtype=int), np.array(srf_spc_idx, dtype=int), srfs)


def wins_from_srfs(srfs, srf_idxs, srf_spc_idxs, srf_mod_idxs):
    """Retrieve windows and associated indices from list of srfs."""

    win_srf_idx, win_spc_idx, win_mod_idx, wins = [], [], [], []

    _zip = zip(srfs, srf_idxs, srf_spc_idxs, srf_mod_idxs)
    for srf, srf_idx, spc_idx, mod_idx in _zip:

        _wins = srf._apertures
        if len(_wins) > 0:
            _len_wins = len(_wins)
            win_srf_idx.extend([srf_idx] * _len_wins)
            win_spc_idx.extend([spc_idx] * _len_wins)
            win_mod_idx.extend([mod_idx] * _len_wins)
            wins.extend(_wins)

    return (np.array(win_mod_idx, dtype=int), np.array(win_spc_idx, dtype=int),
            np.array(win_srf_idx, dtype=int), wins)


def perim_spcs_mask(spcs):
    """Perimeter mask for list of spcs.

    Args:
        spcs: List of spcs from Dragonfly building model.
    Returns:
        Boolean mask of perimeter spc index.
    """

    perim_mask = np.zeros(len(spcs))

    for i, room in enumerate(spcs):
        if room.exterior_wall_area > 1e-10:
            perim_mask[i] = 1

    return perim_mask


def perim_srfs_mask(srfs):
    """Perimeter mask for bem surfaces (perimeter walls, floors, and roofceilings).

    Args:
        srfs: List of faces from Honeybee rooms.

    Returns:
        Boolean mask of perimeter surface index.
    """

    perim_mask = np.zeros(len(srfs))

    for i, face in enumerate(srfs):

        _is_int_wall = \
            isinstance(face.type, facetype.Wall) and \
            (isinstance(face.boundary_condition, bc.Surface) or
                isinstance(face.boundary_condition, bc.Adiabatic))
        if not _is_int_wall:
            perim_mask[i] = 1

    return perim_mask


def simulate_batch(batch_fpath, epw_abs_fpath, model_json_fnames, sim_abs_fpath=None):
    """Write batch file for simulation.

    Simulation are run with dragonfly-energy. Install dragonfly-energy with:

    .. code-block:: console
        >> pip install -U dragonfly-energy
        >> pip install dragonfly-energy==1.10.30  # stable version

    A single model can then be simulated with:

    .. code-block:: console
        >> dragonfly-energy simulate model <path-to-model-json> <absolute-path-to-epw>

    Batches are run with (in git bash):

    .. code-block:: console
        >> . json_batch.sh

    WARNINGS:

    Running VSCode in Remote-WSL mode will lock file permissions which will prevent
    simulations from running. The solution is to close the VSCode window while running
    simulations.

    Run in git bash rather then Ubuntu bash because Honeybee's OpenStudio checks doesn't
    seem to work in WSL.

    Args:
        batch_fpath: file path to generate batch file.
        epw_abs_fpath: file path to epw. Must be absolute so that it can be used by
            openstudio.exe.
        model_json_fnames: List of model_json file names.
        sim_fpath: Path to simulation parameter json.
    """

    sim_cmd = '--sim-par-json "{}"'.format(
        sim_abs_fpath) if sim_abs_fpath else ''
    # --bypass-check
    with open(batch_fpath, 'w') as fp:
        for name in model_json_fnames:
            fp.write('dragonfly-energy simulate model "{}" "{}" {} '
                     '\n'.format(name, epw_abs_fpath, sim_cmd))
