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
    binstat = binned_statistic(qs, qs, bins=3, statistic="mean")
    df = pmf._make_pmf1(qs, ps, binstat)

    # Correct values
    bin_idx = [0, 1, 2]
    ps = np.array([5, 2, 2]) / 9.0

    # Check
    assert np.allclose(df.index.values, bin_idx, atol=1e-10)
    assert np.allclose(ps, ps, atol=1e-10)

def test_make_bin_arr():
    """Test _make_hist1."""

    data = _make_dataset()
    qs, ps = data.qs.values, data.ps.values
    binstat = binned_statistic(qs, qs, bins=3, statistic="mean")

    tst_binned_qs = pmf.make_bin_arr(qs, binstat.binnumber - 1)
    tst_binned_ps = pmf.make_bin_arr(ps, binstat.binnumber - 1)

    # Correct values
    binned_qs = [[.1, .1], [.2, .2], [.3]]
    binned_ps = [[2., 3.], [1., 1.], [2.]]

    assert len(binned_qs) == len(tst_binned_qs)
    assert len(binned_ps) == len(tst_binned_ps)
    assert [np.allclose(tst_qs, qs, atol=1e-10)
            for tst_qs, qs in zip(tst_binned_qs, binned_qs)]
    assert [np.allclose(tst_ps, ps, atol=1e-10)
            for tst_ps, ps in zip(tst_binned_ps, binned_ps)]


if __name__ == "__main__":
    test_make_pmf1()
    test_make_hist1()