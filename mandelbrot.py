import torch
import numpy as np
import matplotlib.pyplot as plt

def compute_mandelbrot(C, max_steps=100, bound=2, power=2, area_factor=9):
    """
    Computes the Mandelbrot set using the standard iteration method

    Parameters:
    - C: ndarray of complex numbers.
    - max_steps: Maximum number of iterations.
    - bound: Divergence threshold.
    - power: Power used in the Mandelbrot calculation.
    - area_factor: Factor for estimating the set area.

    Returns:
    - num_div_steps: 2D array of the number of steps before divergence.
    - area_at_step: list of the estimated area at each iteration.
    """
    Zi = C.copy()
    num_div_steps = np.zeros_like(C, dtype=np.int64)
    area_at_step = []
    for i in range(max_steps):
        Zi = np.power(Zi, power) + C
        num_div_steps += (np.abs(Zi) < bound)

        rel_area = np.sum(np.abs(Zi) < bound) / np.prod(C.shape)
        area_at_step.append(rel_area * area_factor)

    print(f"Final area estimation: {area_at_step[-1]}")

    return num_div_steps, np.array(area_at_step)


def compute_mandelbrot_torch(C, max_steps=100, bound=2, power=2, area_factor=9):
    """
    Computes the Mandelbrot set using the standard iteration method
    Uses pytorch methods instead of numpy for better scaling on multple processors / gpu

    Parameters:
    - C: ndarray of complex numbers.
    - max_steps: Maximum number of iterations.
    - bound: Divergence threshold.
    - power: Power used in the Mandelbrot calculation.
    - area_factor: Factor for estimating the set area.

    Returns:
    - num_div_steps: 2D array of the number of steps before divergence.
    - area_at_step: list of the estimated area at each iteration.
    """
    if not torch.is_tensor(C):
      C = torch.from_numpy(C)
    num_samples = C.numel()
    Zi = C.detach().clone()
    num_div_steps = torch.zeros_like(C, dtype=torch.int64)
    area_at_step = []
    for i in range(max_steps):
        Zi = torch.pow(Zi, power) + C
        num_div_steps += (torch.abs(Zi) < bound)

        rel_area = torch.sum(torch.abs(Zi) < bound) / num_samples
        area_at_step.append(rel_area * area_factor)

    print(f"Final area estimation: {area_at_step[-1]}")

    return num_div_steps, torch.tensor(area_at_step)

