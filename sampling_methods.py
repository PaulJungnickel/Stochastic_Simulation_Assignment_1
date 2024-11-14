import numpy as np


def generate_complex_grid(resolution, real_range=(-2, 1), imag_range=(-1.5, 1.5)):
    rval = np.linspace(real_range[0], real_range[1], resolution).reshape(1, -1)
    ival = np.linspace(imag_range[0], imag_range[1], resolution).reshape(-1, 1)
    C = rval + 1.j * ival
    return C


def uniform_random_sampling(num_samples, real_range=(-2, 2), imag_range=(-2, 2), seed=42):
    np.random.seed(seed)
    rval = np.random.uniform(real_range[0], real_range[1], num_samples)
    ival = np.random.uniform(imag_range[0], imag_range[1], num_samples)
    return rval + 1.j * ival