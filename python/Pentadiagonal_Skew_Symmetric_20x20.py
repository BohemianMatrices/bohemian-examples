#!/usr/bin/env python3

from bohemian import Bohemian
from bohemian.matrix_generators import get_dtype
from bohemian.histogram_maps import inverse_density
import matplotlib.pyplot as plt
import numpy as np


def random_pentadiagonal_skew_symmetric_matrix(population, n, d=None):

    # Get the datatype to use for output
    dt = get_dtype(population, d)

    def random_pentadiagonal_skew_symmetric_matrix(batch_size):

        _dt = dt
        _population = population
        _n = n
        _d = d

        while True:

            if _d is None:
                v_main = np.random.choice(_population, (batch_size, _n)).astype(
                    _dt)
                A_main = np.apply_along_axis(np.diag, 1, v_main)
            else:
                A_main = np.apply_along_axis(np.diag, 1, _d * np.ones(
                    (batch_size, _n))).astype(_dt)

            v = np.random.choice(_population, (batch_size, _n - 2))
            A_sub = np.apply_along_axis(np.diag, 1, v, k=-2)
            A_super = np.apply_along_axis(np.diag, 1, -v, k=2)

            A = A_main + A_sub + A_super

            yield A

    return random_pentadiagonal_skew_symmetric_matrix

P = np.exp(1j * np.pi * np.array([-2/6, -1/6, 0, 1/6, 2/6]))
g = random_pentadiagonal_skew_symmetric_matrix(P, 20, d=0)

# Initialize the Bohemian object
bhime = Bohemian(generator = g,
                 matrices_per_file = 10**6,
                 verbose = 10)

# Compute eigenvalues
bhime.compute_eigenvalues(num_files = 1)

# Histogram
histogram_filename = bhime.generate_histogram(height = 2001,
                                              axis_range = [[-2, 2], [-2, 2]],
                                              symmetry_imag = True,
                                              symmetry_real = True)

# Final image color
bhime.plot(histogram_file = histogram_filename,
           cm = plt.cm.viridis,
           histogram_map = inverse_density)
