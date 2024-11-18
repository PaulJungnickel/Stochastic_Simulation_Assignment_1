import numpy as np
from scipy.stats.qmc import LatinHypercube, scale, Sobol

def generate_complex_grid(resolution, real_range=(-2, 1), imag_range=(-1.5, 1.5)):
    """
    Generates a grid of complex numbers for the Mandelbrot set.

    Parameters:
    - resolution: Number of points along each axis.
    - real_range: Tuple specifying the range for the real axis.
    - imag_range: Tuple specifying the range for the imaginary axis.

    Returns:
    - C: 2D array of complex numbers representing the grid.
    """
    # Create equally spaced values for the real and imaginary parts
    rval = np.linspace(real_range[0], real_range[1], resolution).reshape(1, -1)
    ival = np.linspace(imag_range[0], imag_range[1], resolution).reshape(-1, 1)
    # Combine real and imaginary parts into complex numbers
    C = rval + 1.j * ival
    return C

def uniform_random_sampling(num_samples, real_range=(-2, 2), imag_range=(-2, 2), seed=42):
    """
    Performs uniform random sampling for the Mandelbrot set.

    Parameters:
    - num_samples: Number of samples.
    - real_range: Tuple specifying the range for the real axis.
    - imag_range: Tuple specifying the range for the imaginary axis.
    - seed: Random seed for reproducibility.

    Returns:
    - C: 1D array of complex numbers representing the samples.
    """
    # Seed the random generator for reproducibility
    np.random.seed(seed)
    # Generate random real and imaginary parts
    rval = np.random.uniform(real_range[0], real_range[1], num_samples)
    ival = np.random.uniform(imag_range[0], imag_range[1], num_samples)
    return rval + 1.j * ival

def orthogonal_sampling(num_samples, real_range=(-2, 1), imag_range=(-1.5, 1.5), seed=42):
    """
    Performs orthogonal sampling for the Mandelbrot set, suited for Python while following the logic of the provided C code.

    Parameters:
    - num_samples: Total number of samples to generate (must be a perfect square).
    - real_range: Tuple specifying the range for the real axis.
    - imag_range: Tuple specifying the range for the imaginary axis.
    - seed: Random seed for reproducibility.

    Returns:
    - C: 1D array of complex numbers representing the samples.
    """
    # Ensure num_samples is a perfect square for orthogonal grid
    np.random.seed(seed)
    major = int(np.round(np.sqrt(num_samples)))
    num_samples = major * major
    print(f"Adjusted num_samples to {num_samples} to ensure it is a perfect square close to the original value.")

    x_indices = np.arange(major)
    y_indices = np.arange(major)

    samples = []
    for i in range(major):
        # Shuffle indices for randomized sampling
        np.random.shuffle(x_indices)
        np.random.shuffle(y_indices)

        for j in range(major):
            # Generate random perturbations within each grid cell
            rand_real = np.random.uniform(0, 1)
            rand_imag = np.random.uniform(0, 1)

            # Map to the real and imaginary ranges
            x = real_range[0] + (real_range[1] - real_range[0]) * ((i + rand_real) / major)
            y = imag_range[0] + (imag_range[1] - imag_range[0]) * ((j + rand_imag) / major)

            samples.append(complex(x, y))

    return np.array(samples)

def latin_hypercube_sampling(num_samples, real_range=(-2, 1), imag_range=(-1.5, 1.5), seed=42):
    """
    Performs Latin Hypercube sampling for the Mandelbrot set.

    Parameters:
    - num_samples: Number of samples.
    - real_range: Tuple specifying the range for the real axis.
    - imag_range: Tuple specifying the range for the imaginary axis.
    - seed: Random seed for reproducibility.

    Returns:
    - C: 1D array of complex numbers representing the samples.
    """
    # Generate Latin Hypercube samples
    np.random.seed(seed)
    sampler = LatinHypercube(d=2)
    lhs_samples = sampler.random(n=num_samples)

    # Scale samples to the desired ranges
    scaled_samples = scale(lhs_samples, [real_range[0], imag_range[0]], [real_range[1], imag_range[1]])
    rval, ival = scaled_samples[:, 0], scaled_samples[:, 1]
    return rval + 1.j * ival

def importance_sampling(num_samples, max_iter, real_range=(-2, 1), imag_range=(-1.5, 1.5), seed=None):
    """
    Performs importance sampling for estimating the area of the Mandelbrot set.

    Parameters:
    - num_samples: Number of samples.
    - max_iter: Maximum number of iterations for Mandelbrot membership.
    - real_range: Tuple specifying the range for the real axis.
    - imag_range: Tuple specifying the range for the imaginary axis.
    - seed: Random seed for reproducibility.

    Returns:
    - weighted_area_estimate: Estimated area of the Mandelbrot set.
    """
    if seed is None:
        seed = random.randint(0, 10000)
    np.random.seed(seed)

    # Generate normally distributed samples
    mean_real = 0
    mean_imag = 0
    std_dev = 0.8
    
    real_samples = np.random.normal(mean_real, std_dev, num_samples)
    imag_samples = np.random.normal(mean_imag, std_dev, num_samples)

    # Clip samples to remain within specified ranges
    real_samples = np.clip(real_samples, real_range[0], real_range[1])
    imag_samples = np.clip(imag_samples, imag_range[0], imag_range[1])

    # Combine into complex numbers
    complex_samples = real_samples + 1j * imag_samples

    # Compute proposal and target densities
    proposal_density = (1 / (2 * np.pi * std_dev**2)) * np.exp(-(real_samples**2 + imag_samples**2) / (2 * std_dev**2))
    target_density = 1 / ((real_range[1] - real_range[0]) * (imag_range[1] - imag_range[0]))

    # Determine if samples are in the Mandelbrot set
    in_set = np.array([is_in_mandelbrot(c, max_iter) for c in complex_samples])

    # Compute weights and estimate area
    weights = target_density / proposal_density
    rect_area = (real_range[1] - real_range[0]) * (imag_range[1] - imag_range[0])
    weighted_area_estimate = np.mean(in_set * weights) * rect_area

    return weighted_area_estimate

def sobol_sampling(num_samples, real_range=(-2, 1), imag_range=(-1.5, 1.5), seed=None):
    """
    Performs Sobol sampling (Quasi-Monte Carlo) for the Mandelbrot set.

    Parameters:
    - num_samples: Number of samples.
    - real_range: Tuple specifying the range for the real axis.
    - imag_range: Tuple specifying the range for the imaginary axis.
    - seed: Random seed for reproducibility.

    Returns:
    - C: 1D array of complex numbers representing the samples.
    """
    if seed is None:
        seed = random.randint(0, 10000)
    np.random.seed(seed)

    # Generate Sobol samples
    sampler = Sobol(d=2, scramble=True)
    sobol_samples = sampler.random(n=num_samples)

    # Scale samples to the desired ranges
    scaled_samples = scale(sobol_samples, [real_range[0], imag_range[0]], [real_range[1], imag_range[1]])

    rval, ival = scaled_samples[:, 0], scaled_samples[:, 1]
    return rval + 1.j * ival

def adaptive_sampling(num_samples, max_iter, real_range=(-2, 1), imag_range=(-1.5, 1.5), seed=None, iterations=10):
    """
    Performs adaptive sampling for estimating the area of the Mandelbrot set.

    Parameters:
    - num_samples: Number of initial samples.
    - max_iter: Maximum number of iterations for Mandelbrot membership.
    - real_range: Tuple specifying the range for the real axis.
    - imag_range: Tuple specifying the range for the imaginary axis.
    - seed: Random seed for reproducibility.
    - iterations: Number of adaptive sampling iterations.

    Returns:
    - area_estimate: Estimated area of the Mandelbrot set.
    """
    if seed is None:
        seed = random.randint(0, 10000)
    np.random.seed(seed)

    # Generate initial samples
    rval = np.random.uniform(real_range[0], real_range[1], num_samples)
    ival = np.random.uniform(imag_range[0], imag_range[1], num_samples)
    complex_samples = rval + 1.j * ival

    # Determine if samples are in the Mandelbrot set
    in_set = np.array([is_in_mandelbrot(c, max_iter) for c in complex_samples])
    rect_area = (real_range[1] - real_range[0]) * (imag_range[1] - imag_range[0])
    area_estimate = np.mean(in_set) * rect_area

    # Adaptive refinement loop
    for _ in range(iterations):
        num_refine_samples = int(num_samples * 0.5)
        refine_rval = np.random.uniform(real_range[0], real_range[1], num_refine_samples)
        refine_ival = np.random.uniform(imag_range[0], imag_range[1], num_refine_samples)
        refine_samples = refine_rval + 1.j * refine_ival

        complex_samples = np.concatenate((complex_samples, refine_samples))
        in_set = np.array([is_in_mandelbrot(c, max_iter) for c in complex_samples])
        area_estimate = np.mean(in_set) * rect_area

    return area_estimate

def pure_random_sampling(num_samples, real_range=(-2, 1), imag_range=(-1.5, 1.5), seed=42):
    """
    Performs pure random sampling for the Mandelbrot set.

    Parameters:
    - num_samples: Number of samples.
    - real_range: Tuple specifying the range for the real axis.
    - imag_range: Tuple specifying the range for the imaginary axis.
    - seed: Random seed for reproducibility.

    Returns:
    - C: 1D array of complex numbers representing the samples.
    """
    np.random.seed(seed)
    # Generate uniformly distributed samples
    rval = np.random.uniform(real_range[0], real_range[1], num_samples)
    ival = np.random.uniform(imag_range[0], imag_range[1], num_samples)
    return rval + 1.j * ival