#!/usr/bin/env python3

from bohemian import Bohemian
from bohemian.matrix_generators import random_upper_hessenberg_toeplitz_matrix
from bohemian.histogram_maps import inverse_density
import matplotlib.pyplot as plt

g = random_upper_hessenberg_toeplitz_matrix([-1, 0, 1], 16, s = 1, d = 0)

# Initialize the Bohemian object
bhime = Bohemian(generator = g,
                 matrices_per_file = 10**6,
                 verbose = 10)

# Compute eigenvalues
bhime.compute_eigenvalues(num_files = 10)

# Histogram
# range = [[xmin, xmax], [ymin, ymax]]
histogram_filename = bhime.generate_histogram(height = 8001,
                                              axis_range = [[-3, 3], [-3, 3]],
                                              symmetry_imag = False,
                                              symmetry_real = False)

# Final image color
from matplotlib.colors import LinearSegmentedColormap
cdict = {'red':   [[0.0,   0.0,  0.0],
                   [0.1,   0.0,  0.0],
                   [0.16,  0.0,  0.0],
                   [0.22,  0.0,  0.0],
                   [0.28,  1.0,  1.0],
                   [0.34,  1.0,  1.0],
                   [0.4,   1.0,  1.0],
                   [0.55,  1.0,  1.0],
                   [1.0,   1.0,  1.0]],
         'green': [[0.0,   0.0,  0.0],
                   [0.1,   0.0,  0.0],
                   [0.16,  1.0,  1.0],
                   [0.22,  1.0,  1.0],
                   [0.28,  1.0,  1.0],
                   [0.34,  0.0,  0.0],
                   [0.4,   0.0,  0.0],
                   [0.55,  1.0,  1.0],
                   [1.0,   1.0,  1.0]],
         'blue':  [[0.0,   0.0,  0.0],
                   [0.1,   1.0,  1.0],
                   [0.16,  1.0,  1.0],
                   [0.22,  0.0,  0.0],
                   [0.28,  0.0,  0.0],
                   [0.34,  0.0,  0.0],
                   [0.4,   0.0,  0.0],
                   [0.55,  1.0,  1.0],
                   [1.0,   1.0,  1.0]]}
cm = LinearSegmentedColormap('testCmap', segmentdata=cdict, N=1024)
bhime.plot(histogram_file = histogram_filename,
           cm = cm)
