import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
import os
#import cupy as cp  # For GPU acceleration

class MandelbrotAnalysis:
    def __init__(self, real_range, imag_range):
        self.real_range = real_range
        self.imag_range = imag_range

    def get_sample_name(self, sample_type):
        sample_name_list = {
            0: "Pure",
            1: "LHS",
            2: "Ortho"
        }
        return sample_name_list.get(sample_type, "Unknown")


    # Sampling methods
    # with real_range and imag_range as the range of the Mandelbrot set
    def pure_random_sampling(self, num_samples):
        return

    def latin_hypercube_sampling(self, num_samples) -> np.ndarray:
        """
        Generate N samples using Latin Hypercube Sampling using an grid like.
        This setup ensures each variable is evenly sampled across its range.
        We assume that the number of dimensions is 2 and 
        that we are sampling for each dimension.

        Parameters
        ----------
        N : int
            Number of samples to generate.
        bounds : tuple
            Lower and upper bounds for the samples.
        
        Returns
        -------
        samples : np.ndarray
            N samples generated using Latin Hypercube Sampling.
        """
        sampler = qmc.LatinHypercube(d=1)
        x_samples = sampler.random(n=num_samples)
        y_samples = sampler.random(n=num_samples)

        x_samples = qmc.scale(x_samples, self.real_range[0], self.real_range[1])
        y_samples = qmc.scale(y_samples, self.imag_range[0], self.imag_range[1])

        samples = np.column_stack((x_samples, y_samples))
        return samples

    def orthogonal_sampling(self, num_samples):
        return

    # Mandelbrot convergence check for real + imag*(1j) with max_iter
    def mandel_convergence_check(self, real, imag, max_iter):
        c = complex(real, imag)
        z = 0
        for i in range(max_iter):
            z = z * z + c
            if abs(z) > 2:
                return False
        return True

    def mandel_convergence_check_vectorized(self, samples, max_iter):
        # complex number array
        c = samples[:, 0] + 1j * samples[:, 1]
        z = np.zeros(c.shape, dtype=np.complex128)
        mask = np.ones(c.shape, dtype=bool)
        
        # we use numpy bool array to keep track of the convergence
        # if the complex number is divergent, we set the corresponding element in the mask to False
        for _ in range(max_iter):
            z[mask] = z[mask] ** 2 + c[mask]

            # create a another bool array to keep track of the divergent complex numbers
            # kick out the divergent complex numbers from the mask
            mask[np.abs(z) > 2] = False 
        
        return mask

    # Core function Monte Carlo estimate of the Mandelbrot set area
    # sequential version
    def monte_c_estimate(self, samples, max_iter):
        count = 0
        for real, imag in samples:
            if self.mandel_convergence_check(real, imag, max_iter):
                count += 1
        area_estimate = count / len(samples) * (self.real_range[1] - self.real_range[0]) * (self.imag_range[1] - self.imag_range[0])
        return area_estimate
    
    # GPU version
    def monte_c_estimate_gpu(self, samples, max_iter):
        return

    # TODO: Design the following metrics and comparison functions
    # Color the Mandelbrot set with plotting the samples
    def color_mandelbrot(self, samples, max_iter, sample_type = 1):
        # check the sampe type
        sample_name = self.get_sample_name(sample_type)

        # get the mask of the samples that are inside the Mandelbrot set
        mask = self.mandel_convergence_check_vectorized(samples, max_iter)

        # plot the samples
        plt.figure(figsize=(10, 10))
        plt.scatter(samples[mask, 0], samples[mask, 1], color='black', s=0.5, label="Inside Mandelbrot Set")
        plt.scatter(samples[~mask, 0], samples[~mask, 1], color='red', s=0.5, alpha=0.6, label="Outside Mandelbrot Set")
        plt.xlabel('Real Axis')
        plt.ylabel('Imaginary Axis')
        plt.title('Visualization of Mandelbrot Set')
        plt.legend()
        
        # store the image into a file, if no existing directory, create one        
        output_dir = '../images/color_mandelbrot'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/mandelbrot_{sample_name}_{len(samples)}_maxIter_{max_iter}.png')

    def metrics(self, monte_c_estimate_fn, baseline_samples, min_iter, max_iter):
        return

    def compare_sampling_methods(self, num_samples, min_iter, max_iter):
        # Code to test the sampling methods, can be removed later.
        pure_random_samples = self.latin_hypercube_sampling(num_samples)
        x, y = pure_random_samples[:, 0], pure_random_samples[:, 1]
        plt.scatter(x, y)
        plt.title('Latin Hypercube Sampling (2D Projection)')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.show()
        return

    def plot_mandelbrot(self, num_samples, min_iter, max_iter):
        return

if __name__ == "__main__":
    mandelbrot = MandelbrotAnalysis(real_range=(-2, 1), imag_range=(-1.5, 1.5))
    num_samples = 1000
    min_iter = 10
    max_iter = 100
    mandelbrot.compare_sampling_methods(num_samples, min_iter, max_iter)
    
    