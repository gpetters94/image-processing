import math
import random

from numba import jit

@jit
def dot(a, b):
    s = 0

    for i in range(0, len(a)):
        s += (a[i] * b[i])

    return s

def train(test_data, e, r, max_iterations, debug, w):

    s = len(test_data)

    for (d, o) in test_data:
        d = [1]+d

    # Initialize weights
    if w == None:
        w = [random.uniform(-1, 1) for _ in test_data[0][0]]

    snorm = s * len(w)

    @jit
    def train():
        err = 0

        for (x, d) in test_data:

            yt = dot(w, x)

            for i in range(0, len(w)):
                w[i] += (r * (d - yt) * x[i])/snorm
                err += abs(d - yt)

        return (err/s)

    n = 0
    while True:

        err = train()

        n += 1
        if debug:
            print("%d: %f" %(n, err))

        if err < e or (max_iterations > 0 and n >= max_iterations):
            return w

def test(data, weights):
    return 1 if dot(data, weights) > 0 else -1
