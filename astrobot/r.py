from functools import reduce
import numpy as np


class R(object):
    """Class to structures hierarchical vector subspaces as dataframes.

    R is a reference to the set of ordered n-tuples of real numbers in n-space denoted R^n.

    Args:
        dfparam: Pandas DataFrame
    """

    def __init__(self, dataframe):

        # Dataframes
        self._mod = None
        self._spc = None
        self._srf = None

        self.mod = dataframe

    def __repr__(self):
        modcols = self.mod.columns.tolist() if self._mod is not None else []
        spccols = self.spc.columns.tolist() if self._spc is not None else []
        srfcols = self.srf.columns.tolist() if self._srf is not None else []
        wincols = self.win.columns.tolist() if self._win is not None else []
        return "mod: {}\nspc: {}\nsrf: {}\nwin: {}".format(
            modcols, spccols, srfcols, wincols)

    @property
    def mod(self):
        return self._mod

    @mod.setter
    def mod(self, mod):
        if self._mod is None:
            mod['idx'] = mod.index
            self._mod = mod.set_index('idx')
        else:
            raise Exception('mod is already set.')

    @property
    def spc(self):
        return self._spc

    @spc.setter
    def spc(self, spc):
        spc['idx'] = spc.index
        self._spc = spc.set_index('idx')

    @property
    def srf(self):
        return self._srf

    @srf.setter
    def srf(self, srf):
        srf['idx'] = srf.index
        self._srf = srf.set_index('idx')

    @property
    def win(self):
        return self._win

    @win.setter
    def win(self, win):
        win['idx'] = win.index
        self._win = win.set_index('idx')
