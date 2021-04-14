import os
import pandas as pd
import numpy as np
from .util import *


EIA_DIR = "resources/eia"
CBECS_SURVEY_PKL = "cbecs_survey.pkl"
CBECS_KEY_PKL = "cbecs_key.pkl"
RECS_SURVEY_PKL = "recs_survey.pkl"
RECS_KEY_PKL = "recs_key.pkl"


class EnergySurvey(object):
    """
    For EIA surevy data.
    """

    @staticmethod
    def ldlst():
        return ['cool', 'heat', 'mech_vent', 'light_electric', 'shw', 'equip_electric', 'misc']

    @staticmethod
    def eldlst():
        return ['eui'] + EnergySurvey.ldlst()

    @staticmethod
    def com_survey_df():
        return pd.read_pickle(os.path.join(ROOT_DIR, EIA_DIR, CBECS_SURVEY_PKL))

    @staticmethod
    def com_key_df():
        return pd.read_pickle(os.path.join(ROOT_DIR, EIA_DIR, CBECS_KEY_PKL))

    @staticmethod
    def res_survey_df():
        return pd.read_pickle(os.path.join(ROOT_DIR, EIA_DIR, RECS_SURVEY_PKL))

    @ staticmethod
    def res_key_df(note=True):
        if note is True:
            recs_note = os.path.join(ROOT_DIR, EIA_DIR, "recs_note.txt")
            f = open(recs_note)
            notes = f.readlines()
            f.close()
            for ln in notes[1:]:
                print(ln[:-1])

        return pd.read_pickle(os.path.join(ROOT_DIR, EIA_DIR, RECS_KEY_PKL))

    @ staticmethod
    def chk_sqft(df):
        return ft.reduce(lambda a, b: not(np.isnan(a) and np.isnan(b)), df.sqft.values)

    @ staticmethod
    def load_breakdown(df):
        def ntn(arr): return np.nan_to_num(arr)
        df['heat'] = ntn(df.mfhtbtu) / df.sqft     # heating
        df['cool'] = ntn(df.mfclbtu) / df.sqft     # cooling
        df['mech_vent'] = ntn(df.mfvnbtu) / df.sqft     # ventilation
        df['light_electric'] = ntn(df.mfltbtu) / df.sqft    # lighting
        df['shw'] = ntn(df.mfwtbtu) / df.sqft      # water heat
        # cooking, refridge, office, pc
        df['equip_electric'] = df[['mfckbtu', 'mfrfbtu',
                                   'mfofbtu', 'mfpcbtu']].sum(axis=1) / df.sqft
        df['misc'] = ntn(df.mfotbtu) / df.sqft      # misc use

        return df

    @ staticmethod
    def load_breakdown_pct(df):
        eui = df['eui']
        df['pct_heat'] = df.heat / eui          # heating
        df['pct_cool'] = df.cool / eui          # cooling
        df['pct_mech_vent'] = df.mech_vent / eui          # ventilation
        df['pct_light_electric'] = df.light_electric / eui        # lighting
        df['pct_shw'] = df.shw / eui            # water heat
        df['pct_equip_electric'] = df.equip_electric / \
            eui        # cooking, refridge, office, pc
        df['pct_misc'] = df.misc / eui          # misc use

        return df

    @ staticmethod
    def loads(df):
        df['eui'] = np.nan_to_num(df.mfbtu) / df.sqft           # eui
        df = EnergySurvey.load_breakdown(df)
        df = EnergySurvey.load_breakdown_pct(df)

        return df
