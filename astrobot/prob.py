import numpy as np
from functools import reduce

def nCr(n:int, r:int) -> int:
    """Number of combinations without repetitions.

    nCr = n! / (n-r)! r!

    Args:
        n: Number of elements.
        r: Length of combinations.

    Returns:
        Number of combinations as int.
    """

    # Check domain
    assert 0 <= r <= n, \
        f"Domain error: 0 <= r:{r} <= n:{n}"

    return (np.prod(range(n, r, -1)) /
            np.prod(range(n-r, 1, -1))).astype(int)


def combinations(items:list, r:int) -> list:
    """Choose r combinations from n items with no repetitions."""

    def _concat(combo:list, items:list, r:int) -> list:

        if len(combo) == r:
            return [combo]

        # Iterate through all available items (item_j) and
        # passes new item to _concat consisting of:
        # combo~combo + items[i] and item~items[j+1:]
        return reduce(
            lambda a, b: a + b,
            [_concat(combo + [items[i]], items[i+1:], r)
             for i, _ in enumerate(items)
             if len(combo) + 1 + len(items[i+1:]) >= r]
        )

    # Check domain
    n = len(items)
    assert 0 <= r <= n, \
        f"Domain error: 0 <= r:{r} <= n:{n}."

    # Pass all initial items to be concatenated.
    combos = _concat([], items, r)

    # Check range
    assert len(combos) == nCr(n, r), \
        f"Error in code. len(combos) != nCr."

    return [tuple(x) for x in combos]
