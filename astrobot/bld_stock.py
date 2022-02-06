import os
import numpy as np
import pandas as pd
from functools import reduce

RESCOMSTOCK_DIR = "C:/users/admin/master/astrobot/resources/rescomstock"

BLD_STOCK_2021_URL = "https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/" + \
    "end-use-load-profiles-for-us-building-stock/2021/"

def osm_url(bld_id, zero_pad=7):
    zeros = reduce(
        lambda a, b: str(a)+"0", range(zero_pad - int(np.log10(bld_id) + 1)))
    return ( BLD_STOCK_2021_URL +
            "comstock_tmy3_release_1/building_energy_models/" +
            "bldg{}-up00.osm.gz".format(zeros + str(bld_id)))


def comstock_df(drop_na=True):
    """Get comstock metadata as dataframe."""
    metacom_fpath = os.path.join(RESCOMSTOCK_DIR, "comstock", "metadata_comstock.tsv")
    comdf = pd.read_csv(metacom_fpath, sep='\t')
    if drop_na:
        comdf = comdf.dropna(axis=0, how="any")
    return comdf


def processed_comstock_df(comdf: pd.DataFrame) -> pd.DataFrame:
    """Simplify comstock dataframe."""

    y_lbls = ["eui"]
    X_lbls = ["cz_cat", "cz", "btype_cat", "btype", "weight", "metadata_index", "bldg_id"]
    comdf[["cz_cat", "btype_cat"]] = comdf[["in.climate_zone_ashrae_2004", "in.building_type"]]
    _mbtu_arr, _sqft_arr = \
        comdf["out.site_energy.total.energy_consumption"], comdf["in.sqft"]
    comdf[y_lbls[0]] = (_mbtu_arr * 1e6 / 3412) / (_sqft_arr / 10.764)  # Convert MBTU, ft2 -> kWh/m2

    # Integer encoding for cz categories
    cz_dict = {cz:i for i, cz in enumerate(np.unique(comdf["cz_cat"]))}
    comdf["cz"] = [cz_dict[cz] for cz in comdf["cz_cat"]]

    # Integer encoding for btype
    btype_dict = {bt:i for i, bt in enumerate(np.unique(comdf["btype_cat"]))}
    comdf["btype"] = [btype_dict[bt] for bt in comdf["btype_cat"]]

    return comdf[X_lbls + y_lbls].copy()