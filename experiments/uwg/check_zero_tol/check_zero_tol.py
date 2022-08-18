
from uwg import Material, Element, Building, BEMDef, SchDef, UWG
import os

def custom_uwg():
    """Generate UWG json with custom reference BEMDef and SchDef objects."""

    # override at 5,2 and add at 18,2

    # SchDef
    default_week = [[0.15] * 24] * 3
    cool_week = [[24 for _ in range(24)] for _ in range(3)]
    heat_week = [[20 for _ in range(24)] for _ in range(3)]

    schdef1 = SchDef(elec=default_week, gas=default_week, light=default_week,
                     occ=default_week, cool=cool_week, heat=heat_week,
                     swh=default_week, q_elec=18.9, q_gas=3.2, q_light=18.9,
                     n_occ=0.12, vent=0.0013, v_swh=0.2846, bldtype='midriseapartment',
                     builtera='new')

    # BEMDedf
    # materials
    insulation = Material(0.049, 836.8 * 265.0, 'insulation')
    gypsum = Material(0.16, 830.0 * 784.9, 'gypsum')
    brick = Material(0.47, 1000000 * 2.018, 'wood')

    # elements
    wall = Element(0.5, 0.92, [0.1, 0.1, 0.0127], [brick, insulation, gypsum], 0, 293, False, 'wood_frame_wall')
    roof = Element(0.7, 0.92, [0.1, 0.1, 0.0127], [brick, insulation, gypsum], 1, 293, True, 'wood_frame_roof')
    mass = Element(0.20, 0.9, [0.5, 0.5], [brick, brick], 0, 293, True, 'wood_floor')

    # building
    bldg = Building(
        floor_height=3.0, int_heat_night=1, int_heat_day=1, int_heat_frad=0.1,
        int_heat_flat=0.1, infil=0.171, vent=0.00045, glazing_ratio=0.4,
        u_value=3.0, shgc=0.3, condtype='AIR', cop=3, coolcap=41, heateff=0.8,
        initial_temp=293)

    bemdef1 = BEMDef(building=bldg, mass=mass, wall=wall, roof=roof,
        bldtype='midriseapartment', builtera='new')

    # vectors
    ref_sch_vector = [schdef1]
    ref_bem_vector = [bemdef1]

    bld = [('midriseapartment', 'new', 1)  # overwrite
           ]  # extend

    #TODO: changed dtsim from 200 to 300.
    model = UWG.from_param_args(
        epw_path=epw_path, bldheight=17.5, blddensity=0.55, vertohor=1.8, zone='5A',
        treecover=0.2, grasscover=0.249, bld=bld, ref_bem_vector=ref_bem_vector,
        ref_sch_vector=ref_sch_vector, month=1, day=1, sensanth=10, nday=89,
        dtsim=200, albroad=0.2)

    return model


if __name__ == "__main__":

    epw_path = os.path.abspath(os.path.join(os.path.curdir, "TUR_Ankara.171280_IWEC.epw"))
    print(os.path.isfile(epw_path))

    # model = UWG.from_param_args(
    #     epw_path=epw_path, bldheight=17.5, blddensity=0.55, vertohor=1.8, zone='5A',
    #     treecover=0.2, grasscover=0.249, month=1, day=1, sensanth=10, nday=89)

    model = custom_uwg()

    model.generate()
    model.simulate()
    # model.write_epw()