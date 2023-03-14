# pylint: skip-file
import numpy as np
from sklearn.preprocessing import normalize
from itertools import combinations, permutations, product, groupby


def countWays(arr, N):
    m = len(arr)
    count = [0 for i in range(N + 1)]

    # base case
    count[0] = 1

    # Count ways for all values up
    # to 'N' and store the result
    for i in range(1, N + 1):
        for j in range(m):
            # if i >= arr[j] then
            # accumulate count for value 'i' as
            # ways to form value 'i-arr[j]'
            if i >= arr[j]:
                count[i] += count[i - arr[j]]

    # required number of ways
    return count[N]


def main():
    # arr = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    # print(countWays(arr, 50))

    # ways = [i  for i in countWays(arr, 50) if len(i) == 5]
    # print(ways)

    traders = ["SNPR", "GDX", "AA", "GVWY", "ZIC", "ZIP", "SHVR"]
    # items2 = items[:4]
    # print(items2)

    combos = list(combinations(traders, 4))
    proportions = [
        [10, 10, 10, 10],
        [20, 10, 5, 5],
        [15, 10, 10, 5],
        [15, 15, 5, 5],
        [25, 5, 5, 5],
    ]
    perms = [p for i in proportions for p in set(permutations(i))]
    session_configs = [list(zip(p[0], p[1])) for p in list(product(combos, perms))]

    print(session_configs)


def normalize_data(x):
    normalized = (x - min(x)) / (max(x) - min(x))
    return normalized


main()
