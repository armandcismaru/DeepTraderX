# pylint: skip-file
from itertools import permutations

proportions2 = [
    [4, 4, 4, 4, 4],
    [8, 4, 4, 4, 0],
    [8, 8, 4, 0, 0],
    [16, 4, 0, 0, 0],
    [20, 0, 0, 0, 0],
    [10, 5, 5, 0, 0],
    [10, 10, 0, 0, 0],
]

proportions = [
    [5, 5, 5, 5, 0],
    [8, 4, 4, 4, 0],
    [8, 8, 2, 2, 0],
    [10, 4, 4, 2, 0],
    [12, 4, 2, 2, 0],
    [14, 2, 2, 2, 0],
    [16, 2, 2, 0, 0],
    [16, 4, 0, 0, 0],
    [18, 2, 0, 0, 0],
    [20, 0, 0, 0, 0],
]
perms = [p for i in proportions for p in set(permutations(i))]
to_append = [0, 0]

# train_traders = [1, 1, 1, 1, 0]
# combos = set(list(permutations(train_traders)))
# session_configs = [list(p) for p in list(product(combos, perms))]
# print(session_configs)

# open csv file to write
with open("markets.csv", "w") as f:
    # write each element of perms to csv file
    for perm in perms:
        f.write(
            f"{perm[0]},{perm[1]},{perm[2]},{perm[3]},{perm[4]},{to_append[0]},{to_append[1]}\n"
        )


# print(perms)
