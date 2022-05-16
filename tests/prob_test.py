
import pytest
import numpy as np
from astrobot import prob
from itertools import combinations as iter_combinations


def test_combinations_base():
    """Test base case of combinations."""

    # Base cases
    # 2c2 = 1
    states = list("AB")
    n, r = len(states), 2
    assert len(prob.combinations(states, r)) == prob.nCr(n, r) == 1

    # 2c0 = [()]; empty set
    states = list("AB")
    n, r = len(states), 0
    combos = prob.combinations(states, r)
    assert len(combos) == prob.nCr(n, r) == 1
    assert combos[0] == ()

    # 2c3 = error
    states = list("AB")
    n, r = len(states), 3
    with pytest.raises(AssertionError):
        prob.combinations(states, r)


def test_combinations():
    """Test combinations"""

    # Non-base cases
    _eqstr = lambda a, b: a == b

    # 5c2
    states = list("ABCDE")
    n, r = len(states), 2
    combos_5c2_ = [
        "AB", "AC", "AD", "AE", #A
            "BC", "BD", "BE", #B
                    "CD", "CE", #C
                        "DE"] #D
    combos_5c2 = prob.combinations(items=states, r=r)
    combos_5c2 = ["".join(x) for x in combos_5c2]
    assert len(combos_5c2) == prob.nCr(n, r)
    assert np.all([_eqstr(x, x_) for x, x_ in zip(combos_5c2, combos_5c2_)])

    # 5c3
    states = list("ABCDE")
    n, r = len(states), 3
    combos_5c3_ = [
        "ABC", "ABD", "ABE",  #A
            "ACD", "ACE",
                    "ADE",
            "BCD", "BCE",  #B
                    "BDE",
                    "CDE"]  #C
    combos_5c3 = prob.combinations(items=states, r=r)
    combos_5c3 = ["".join(x) for x in combos_5c3]
    assert len(combos_5c3) == prob.nCr(n, r)
    assert np.all([_eqstr(x, x_) for x, x_ in zip(combos_5c3, combos_5c3_)])

    # Test redundancies
    items = list("ABCDDD")
    n, r = len(items), 3
    combos = ["".join(x) for x in prob.combinations(items, r)]
    combos_ = ["".join(x) for x in iter_combinations(items, r)]
    assert len(combos) == prob.nCr(n, r)
    assert np.all([_eqstr(x, x_) for x, x_ in zip(combos, combos_)])

    # 7c5
    states = list("ABCDEFG")
    n, r = len(states), 5
    combos = prob.combinations(states, r=r)
    combos = ["".join(x) for x in combos]
    assert len(combos) == prob.nCr(n, r)

    # 2c1
    states = list("AB")
    n, r = len(states), 1
    combos = prob.combinations(states, r=r)
    combos = ["".join(x) for x in combos]
    assert len(combos) == prob.nCr(n, r)

    # 10c9
    states = list("0123456789")
    n, r = len(states), 9
    combos = prob.combinations(states, r=r)
    assert len(combos) == prob.nCr(n, r)

    # 10c6
    states = list("0123456789")
    n, r = len(states), 6
    combos = prob.combinations(states, r=r)
    assert len(combos) == prob.nCr(n, r)
