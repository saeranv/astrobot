"""Distributions from empirical datasets"""
import numpy as np
import pandas as pd
from pprint import pprint
from functools import reduce
from typing import Sequence, Callable
from dataclasses import dataclass
def pp(x, *args): pprint(x) if not args else (pprint(x), pp(*args))


@dataclass
class Pmf(object):
    """One-dimensional Probability Mass Function (pmf) for discrete data."""

    def __init__(self, qs:np.ndarray, bins:int, ps:np.ndarray=None,
                 bin_stat_fx:Callable=None, id:np.ndarray=None) -> None:
        """Initialize pmf properties."""

        # Required args
        self.qs_arr = _seq2arr(qs)
        self.bins = bins
        self._bin_stat = np.mean if bin_stat_fx is None else bin_stat_fx

        # Optional args
        data_num = self.qs_arr.shape[0]
        self.ps_arr = np.ones(data_num) / data_num \
                      if ps is None else _seq2arr(ps)
        self._id_arr = np.arange(data_num, dtype=int) if id is None \
                       else np.array(id, dtype=int)
        self.N = np.sum(self.ps_arr)  # total var dataset size

        # Cached args
        self._bin_edges = None
        self._bin_idx = None
        self._df = None

        # Formatting
        self.prec = 2

    def __repr__(self) -> str:
        return "Pmf: bins={}, N={}.".format(self.bins, round(self.N, 2))

    @property
    def bin_idx(self):
        """Nested list of qs indices per bin.

        qs = [0, 1, 3, 1, 2]
        dist = Pmf(qs, ps, bins=3)
        dist.bin_idx: [[0], [1, 3], [2, 4]]
        """
        if self._bin_idx is None:
            self._bin_idx = bin_idx(self.qs_arr, self.bin_edges)
        return self._bin_idx

    @property
    def bin_edges(self):
        """Array of bin_edges.

        qs = [0, 1, 3, 1, 2]
        dist = Pmf(qs, ps, bins=3)
        dist.bin_edges: [0, 1, 2, 3]
        """
        if self._bin_edges is None:
            self._bin_edges = bin_edges(
                self.qs_arr, bins=self.bins,
                bin_range=(self.qs_arr.min(), self.qs_arr.max()))
        return self._bin_edges

    @property
    def df(self):
        if self._df is None:
            self._df = _pmf(self.qs_arr, self.ps_arr, self.bin_idx,
                self.bin_edges, self._bin_stat)
        return self._df


@dataclass
class JointPmf():
    """Two-dimensional Probability Mass Function (pmf) for discrete data."""

    def __init__(self, pmf1:Pmf, pmf2:Pmf, states1:np.ndarray,
                 states2:np.ndarray) -> None:
        """Initialize pmf properties.

        When defining states1 and states 2, ensure missing bins
                are represented, i.e. states1 = np.unique(qs_arr) may not work.

        Args:
            states1: All unique states in pmf1.
            states2: All unique states in pmf2.
        """

        # Required args
        self.pmf1 = pmf1
        self.pmf2 = pmf2
        # TODO: make this based on bin_edges
        self.states1 = states1
        self.states2 = states2

        # Formatting
        self.prec = 2

        # Optional
        self._jmtx = None
        self._df = None


    @property
    def jmtx(self):
        if self._jmtx is None:
            self._jmtx = _joint_pmf_mtx(
                self.pmf1.qs_arr, self.pmf2.qs_arr, self.states1, self.states2)
        return self._jmtx

    @property
    def df(self):
        if self._df is None:
            jmtx = self.jmtx
            # Make joint Pmf from joint matrix
            _df = pd.DataFrame({"ps": jmtx.reshape(jmtx.size)})
            _df.index = pd.MultiIndex.from_product(iterables=[self.states1, self.states2])
            _df.index = _df.index.set_names(['q1', 'q2'])
            self._df = _df

        return self._df


def _seq2arr(seq:Sequence) -> np.ndarray:
    """Cast to np.ndarray else throw exception."""
    seq = seq if isinstance(seq, np.ndarray) else np.array(seq)
    assert isinstance(seq, np.ndarray), "Input sequence must be " \
        "np.ndarray, got {}.".format(seq)
    return seq


def _pmf(qs:np.ndarray, ps:np.ndarray, bin_idx:Sequence,
               bin_edges:np.ndarray, bin_stat_fx:Callable) -> np.ndarray:
    """Pmf object from empirical data."""

    N = np.sum(ps)  # total var dataset size
    bins = len(bin_idx)

    bin_ps = [np.sum(ps[bi]) for bi in bin_idx]
    bin_mu = [bin_stat_fx(bin_edges[[i-1, i]])
              for i in range(1, bins + 1)]

    # Create initial dataframe
    bin_df = pd.DataFrame(
        {"bins": range(bins),
         "ps": np.array(bin_ps) / N, # normalize by dataset size
         "qs": np.array(bin_mu)})
    bin_df.set_index("bins", inplace=True)
    return bin_df


def _joint_pmf_mtx(qs1:np.ndarray, qs2:np.ndarray, states1:Sequence,
                        states2:Sequence) -> np.ndarray:
    """Joint pmf from empirical dataset."""

    # Init vars
    qs1_num, qs2_num = len(states1), len(states2)
    pxy = np.zeros(shape=(qs1_num, qs2_num))

    # # Outerproduct of states
    states_arr = [(si, sj, state1, state2)
                  for si, state1 in enumerate(states1)
                  for sj, state2 in enumerate(states2)]

    # Check if data1, data2 are in state and return
    # TODO: Move this as categorical_is_in_bin
    # TODO: Add a continous_is_in_bin:
    # i.e hi, lo = bin_edges[var1_state]; return lo <= var1 < hi
    is_intersect_fx = lambda state1, state2, q1, q2: \
        (q1 == state1) and (q2 == state2)

    for si, sj, state1, state2 in states_arr:
        for q1, q2 in zip(qs1, qs2):
            is_intersect = is_intersect_fx(q1, q2, state1, state2)
            pxy[si, sj] += 1 if is_intersect else 0

    return pxy / np.sum(pxy)

def bin_edges(vals:np.ndarray, bins:int, bin_range:Sequence[float]) -> np.ndarray:
    """Create bin_edges."""
    # Get bin_edges
    min_edge, max_edge = bin_range
    bin_inc = (max_edge - min_edge) / bins
    return min_edge + np.array([bin_inc * i for i in range(bins + 1)])


def bin_idx(qs:np.ndarray, bin_edges:np.ndarray, eps=1e-10) -> Sequence:
    """Creates bin_idx, a list of arrays from qs. Arrays are of unequal length."""

    bins = len(bin_edges) - 1
    bin_edges[-1] += eps  # to force last datapoint in

    # Filter by bin_edges
    _in_bin_fx = lambda v, lo, hi: np.where((v >= lo) & (v < hi))
    bin_idx = [_in_bin_fx(qs, bin_edges[i-1], bin_edges[i])
               for i in range(1, bins + 1)]

    # Clean datatype
    return [bi[0].astype(int).tolist() for bi in bin_idx]


def prec(pmf_df: pd.DataFrame, precision:int=2) -> pd.DataFrame:
    """Mutates precisions in Pmf for pretty printing."""
    pmf_df_ = pmf_df.copy()
    pmf_df_.loc[:, "ps"] = np.round(pmf_df_["ps"].values, precision)
    return pmf_df_


# TODO move this to viz
import matplotlib.pyplot as plt

def plt_hist(pmf:Pmf, **kwargs) -> plt.Axes:
    """Plot histogram"""

    _pop_kwarg = lambda q: kwargs.pop(q) if q in kwargs else False
    ax = key if (key := _pop_kwarg("ax")) else plt.subplots()[1]
    edgecolor = x if (x := _pop_kwarg("edgecolor")) else 'black'
    facecolor = x if (x := _pop_kwarg("facecolor")) else "lightgray"


    # TODO: should be sampled, use U(0, 1) => min(q), max(qs) and
    # hashed to appropriarate pmf edge. Also switch to CDF?
    data = [np.ones(int(p * 10000)) * q
            for p, q in zip(pmf.df.ps, pmf.df.qs)]

    return pd.DataFrame(
        {'qs': np.concatenate(data)}).hist(
            bins=pmf.bins, ax=ax, density=True, edgecolor=edgecolor,
            facecolor=facecolor, **kwargs)[0]


def plt_hist2(jpmf: JointPmf, fig_ax=None, **kwargs) -> plt.Axes:
    """Plot 2d histogram for joint distributions."""

    # _pop_kwarg = lambda q: kwargs.pop(q) if q in kwargs else False
    fig, ax = plt.subplots() if fig_ax is None else fig_ax

    im = ax.imshow(jpmf.jmtx, cmap='gray_r', vmin=0, vmax=1)
    fig.colorbar(im);

    return ax