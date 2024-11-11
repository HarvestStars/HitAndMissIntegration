import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

"""
In Latin hypercube sampling one must first decide how many sample
points to use and for each sample point remember in which row and
column the sample point was taken. Such configuration is similar
to having N rooks on a chess board without threatening each other.
"""

def latin_hypercube_sampling(N, bounds) -> np.ndarray:
    """
    Generate N samples using Latin Hypercube Sampling.

    Parameters
    ----------
    N : int
        Number of samples to generate.
]   bounds : tuple
        Lower and upper bounds for the samples.
    
    Returns
    -------
    samples : np.ndarray
        N samples generated using Latin Hypercube Sampling.
    """
    sampler = qmc.LatinHypercube(d=2)
    samples = sampler.random(n=N)
    samples = qmc.scale(samples, bounds[0], bounds[1])
    print(type(samples))
    return samples

if __name__ == "__main__":
    N = 10
    samples = latin_hypercube_sampling(N, (-1, 1))
    x = samples[:, 0]
    y = samples[:, 1]

    plt.scatter(x, y)
    plt.title('Latin Hypercube Sampling (2D Projection)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()