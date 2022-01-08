from astrobot import pmf
import numpy as np
import pandas as pd
from scipy.stats._binned_statistic import binned_statistic


# Tests
def _make_dataset():
    """Make dataset for test."""
    return pd.DataFrame(
        {"qs": np.array([.1, .2, .2, .3, .1]),
         "ps": np.array([2., 1., 1., 2., 3.])})

def test_make_pmf1():
    """Test _make_pmf1."""

    data = _make_dataset()
    qs, ps = data.qs.values, data.ps.values

    bin_idx, bin_mu, _ = pmf.make_bin_data(qs, bins=3, bin_range=(qs.min(), qs.max()))
    df = pmf._make_pmf1(qs, ps, ps, bin_idx, bin_mu)

    # Correct values
    bin_idx = [0, 1, 2]
    ps = np.array([5, 2, 2]) / 9.0

    # Check
    assert np.allclose(df.index.values, bin_idx, atol=1e-10)
    assert np.allclose(ps, ps, atol=1e-10)


def test_make_bin_data():
    """Test binnumber and bin_edges"""

    #     0  1  2  3  4  5  6  7  8
    qs = [1, 1, 2, 3, 2, 5, 3, 1, 6]
    # 1: [1, 1, 1]
    # 2: [2, 2]
    # 3: [3, 3]
    # 4: []
    # 5: [5]
    #    [1, 1, 2, 3, 2, 5, 3, 1, 5]
    bins=5; qs = np.array(qs)
    bin_data = pmf.make_bin_data(qs, bins=bins, bin_range=(qs.min(), qs.max()))
    tst_bin_idx, tst_bin_mu, tst_bin_edges = bin_data

    # Correct data
    binstat = binned_statistic(qs, qs, bins=bins)

    # Test bin_edge
    assert np.allclose(tst_bin_edges, binstat.bin_edges, atol=1e-10)

    # Test bin_mu
    bin_mu = [1.5, 2.5, 3.5, 4.5, 5.5]
    assert np.allclose(tst_bin_mu, bin_mu, atol=1e-10), tst_bin_mu

    # Tests binnumber
    tst_bin_arr = [qs[bi] for bi in tst_bin_idx]
    binnumber_ = binstat.binnumber - 1
    bin_arr = [qs[np.where(binnumber_ == bi)[0]]
               for bi in range(binnumber_.max())]
    tst_bin_idx_ = [np.allclose(tst_bin, bin, atol=1e-10)
                    for tst_bin, bin in zip(tst_bin_arr, bin_arr)]
    assert np.all(tst_bin_idx_)


def test_make_bin_idx():
    """Test _make_idx."""

    data = _make_dataset()
    qs, ps = data.qs.values, data.ps.values
    binstat = binned_statistic(qs, qs, bins=3, statistic="mean")

    bin_idx, _, _ = pmf.make_bin_data(qs, bins=3, bin_range=(qs.min(), qs.max()))
    tst_binned_qs = [qs[bi] for bi in bin_idx]
    tst_binned_ps = [ps[bi] for bi in bin_idx]

    # Correct values
    binned_qs = [[.1, .1], [.2, .2], [.3]]
    binned_ps = [[2., 3.], [1., 1.], [2.]]

    assert len(binned_qs) == len(tst_binned_qs)
    assert len(binned_ps) == len(tst_binned_ps)
    assert [np.allclose(tst_qs, qs, atol=1e-10)
            for tst_qs, qs in zip(tst_binned_qs, binned_qs)]
    assert [np.allclose(tst_ps, ps, atol=1e-10)
            for tst_ps, ps in zip(tst_binned_ps, binned_ps)]
