from datetime import datetime

import numpy as np
from numba import cuda

# eq 14, bits 28-29: IX XI XX
# eq 13, bits 26-27: IY XI XY
# eq 12, bits 23-25: IZ XI XZ
# eq 11, bits 22-23: IX YI YX
# eq 10, bits 20-21: IY YI YY
# eq 09, bits 18-19: IZ YI YZ
# eq 08, bits 16-17: IX ZI ZX
# eq 07, bits 14-15: IY ZI ZY
# eq 06, bits 12-13: IZ ZI ZZ
# eq 05, bits 10-11: XX ZY YZ
# eq 04, bits 08-09: YY XZ ZX
# eq 03, bits 06-07: ZZ YX XY
# eq 02, bits 04-05: XX YY ZZ
# eq 01, bits 02-03: XY YZ ZX
# eq 00, bits 00-01: XZ YX ZY

EQS = [
    ["XZ", "YX", "ZY"],
    ["XY", "YZ", "ZX"],
    ["XX", "YY", "ZZ"],
    ["ZZ", "YX", "XY"],
    ["YY", "XZ", "ZX"],
    ["XX", "ZY", "YZ"],
    ["IZ", "ZI", "ZZ"],
    ["IY", "ZI", "ZY"],
    ["IX", "ZI", "ZX"],
    ["IZ", "YI", "YZ"],
    ["IY", "YI", "YY"],
    ["IX", "YI", "YX"],
    ["IZ", "XI", "XZ"],
    ["IY", "XI", "XY"],
    ["IX", "XI", "XX"],
]

VARS = [
    "IX",
    "IY",
    "IZ",
    "XI",
    "XX",
    "XY",
    "XZ",
    "YI",
    "YX",
    "YY",
    "YZ",
    "ZI",
    "ZX",
    "ZY",
    "ZZ",
]

NUM_CHECKS = 15 * 6

def get_pos(var_name):
    for j, eq in enumerate(EQS):
        for k, var in enumerate(eq):
            if var == var_name:
                yield (j, k)

POSS = tuple(tuple(get_pos(v)) for v in VARS)

def opt_get_bit(use_cuda):
    def get_bit(bstring, num):
        return (bstring >> num) & 1
    if use_cuda:
        return cuda.jit(device=True)(get_bit)
    else:
        return get_bit

def opt_get_var(use_cuda):
    get_bit = opt_get_bit(use_cuda)
    def get_var(bstring, eq_num, bit_num):
        if bit_num == 2:
            temp = get_bit(bstring, eq_num * 2) ^ get_bit(bstring, eq_num * 2 + 1)
            if eq_num < 3:
                temp = temp ^ 1
            return temp
        else:
            return get_bit(bstring, eq_num * 2 + bit_num)
    if use_cuda:
        return cuda.jit(device=True)(get_var)
    else:
        return get_var

def opt_check_var(use_cuda):
    get_var = opt_get_var(use_cuda)
    def check_var(a, b, eq1, v1, eq2, v2, eq3, v3):
        acc = 0
        acc = acc + (get_var(a, eq1, v1) ^ get_var(b, eq2, v2))
        acc = acc + (get_var(a, eq1, v1) ^ get_var(b, eq3, v3))
        acc = acc + (get_var(a, eq2, v2) ^ get_var(b, eq1, v1))
        acc = acc + (get_var(a, eq2, v2) ^ get_var(b, eq3, v3))
        acc = acc + (get_var(a, eq3, v3) ^ get_var(b, eq1, v1))
        acc = acc + (get_var(a, eq3, v3) ^ get_var(b, eq2, v2))
        return acc
    if use_cuda:
        return cuda.jit(device=True)(check_var)
    else:
        return check_var

def opt_check_strategy(use_cuda):
    check_var = opt_check_var(use_cuda)
    def check_strategy(a, b):
        acc = 0
        for v in POSS:
            acc = acc + check_var(a, b, v[0][0], v[0][1], v[1][0], v[1][1], v[2][0], v[2][1])
        return acc
    if use_cuda:
        return cuda.jit(device=True)(check_strategy)
    else:
        return check_strategy

check_strategy = opt_check_strategy(False)
cuda_check_strategy = opt_check_strategy(True)

DIM = 1
BLOCKS_PER_GRID = 256
THREADS_PER_BLOCK = 256

#STRATEGY_RANGE = 1 << 30
STRATEGY_RANGE = 1 << 16
RANGE_PER_THREAD = STRATEGY_RANGE / (THREADS_PER_BLOCK * BLOCKS_PER_GRID)

@cuda.jit
def check_strategies(io):
    pos = cuda.grid(DIM)
    min_a_strat = 0
    min_b_strat = 0
    min_val = NUM_CHECKS
    for j in range(RANGE_PER_THREAD * pos, RANGE_PER_THREAD * (pos + 1)):
        for k in range(STRATEGY_RANGE):
            temp = cuda_check_strategy(j, k)
            if temp < min_val:
                min_a_strat = j
                min_b_strat = k
                min_val = temp
    io[pos][0] = min_a_strat
    io[pos][1] = min_b_strat
    io[pos][2] = min_val

#data = np.zeros((BLOCKS_PER_GRID * THREADS_PER_BLOCK, 3), dtype=int)
#t = datetime.now()
#check_strategies[BLOCKS_PER_GRID, THREADS_PER_BLOCK](data)
#print("Took time", datetime.now() - t)
#temp = data[0]
#for x in data:
#    if x[2] < temp[2]:
#        temp = x
#print(temp)

print(check_strategy(10244, 32840))
print(check_strategy(32840, 10244))
