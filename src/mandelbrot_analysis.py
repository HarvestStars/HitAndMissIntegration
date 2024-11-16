import os
import sys
import ctypes

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

#import cupy as cp  # For GPU acceleration

IMG_COLOR_DIR = '../images/color_mandelbrot'
IMG_CONVERGENCE_DIR = '../images/convergence_analysis'
IMG_CONVERGENCE_IMPROVE_DIR = '../images/convergence_improvement'

class MandelbrotAnalysis:
    def __init__(self, real_range, imag_range):
        self.real_range = real_range
        self.imag_range = imag_range
        self.lib = None

    def _load_library(self):
        # combine the path of the shared library
        lib_path = os.path.join(os.path.dirname(__file__))
        lib_path = os.path.join(lib_path, "..", "ortho-pack", "lib")
        
        print(f"Ready to do Ortho sampling...")
        if sys.platform.startswith("win"):
            print("Windows platform detected.")
            lib_file = "ortho_sampling_generate.dll"
        elif sys.platform.startswith("linux"):
            print("Linux platform detected.")
            lib_file = "libortho_sampling_generate.so"
        elif sys.platform.startswith("darwin"):
            print("MacOS platform detected, wait, what? I have no MacOS.")
            lib_file = "libortho_sampling_generate.dylib" # not implemented, I have no MacOS
        else:
            raise OSError("Unsupported operating system.")

        lib_full_path = os.path.join(lib_path, lib_file)

        # load the shared library
        try:
            self.lib = ctypes.CDLL(lib_full_path)
        except OSError as e:
            raise RuntimeError(f"Unable to load the shared library: {e}")

        # define the function signature
        self.lib.ortho_sampling_generate.argtypes = [
            ctypes.c_int,  # major
            ctypes.c_int,  # runs
            ctypes.c_double, #double min_bound_real,
            ctypes.c_double, #double max_bound_real,
            ctypes.c_double, #double min_bound_imag,
            ctypes.c_double, #double max_bound_imag,
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),  # points_real
            np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')   # points_imag
        ]
        self.lib.ortho_sampling_generate.restype = None

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
        """
        This function implements purey random sampling method using the
        uniform random number generator in the Numpy package
        Input: expected number of samples
        Output: the x and y coordinates of those samples
        """
        
        # improve the code by using numpy vectorization
        rng = np.random.default_rng()
        x_samples = rng.uniform(low=self.real_range[0], high=self.real_range[1], size=num_samples)
        y_samples = rng.uniform(low=self.imag_range[0], high=self.imag_range[1], size=num_samples)
        samples = np.column_stack((x_samples, y_samples))
        return samples

    def pure_random_sampling_partial(self, num_samples, real_min, real_max, imag_min, imag_max):
        rng = np.random.default_rng()
        x_samples = rng.uniform(low=real_min, high=real_max, size=num_samples)
        y_samples = rng.uniform(low=imag_min, high=imag_max, size=num_samples)
        samples = np.column_stack((x_samples, y_samples))
        return samples    

    def latin_hypercube_sampling(self, num_samples) -> np.ndarray:
        """
        Generate N samples using Latin Hypercube Sampling using an grid like.
        This setup ensures each variable is evenly sampled across its range.
        We assume that the number of dimensions is 2 and 
        that we are sampling for each dimension.
        """
        sampler = qmc.LatinHypercube(d=1)
        x_samples = sampler.random(n=num_samples)
        y_samples = sampler.random(n=num_samples)

        x_samples = qmc.scale(x_samples, self.real_range[0], self.real_range[1])
        y_samples = qmc.scale(y_samples, self.imag_range[0], self.imag_range[1])

        samples = np.column_stack((x_samples, y_samples))
        return samples

    def orthogonal_sampling(self, num_samples_root):
        major = num_samples_root  # major is the number of samples in each dimension
        num_samples = major * major  # total number of samples
        runs = 1 # number of runs

        # store the generated points in these arrays
        points_real = np.zeros(num_samples, dtype=np.float64)
        points_imag = np.zeros(num_samples, dtype=np.float64)

        # call the shared library function to generate the points
        self.lib.ortho_sampling_generate(major, runs, self.real_range[0], self.real_range[1], self.imag_range[0], self.imag_range[1], points_real, points_imag)

        # combine the real and imaginary parts to get the samples
        samples = np.column_stack((points_real, points_imag))

        return samples

    def orthogonal_sampling_partial(self, num_samples_root, real_min, real_max, imag_min, imag_max):
        major = num_samples_root
        num_samples = major * major 
        runs = 1
        points_real = np.zeros(num_samples, dtype=np.float64)
        points_imag = np.zeros(num_samples, dtype=np.float64)
        self.lib.ortho_sampling_generate(major, runs, real_min, real_max, imag_min, imag_max, points_real, points_imag)
        samples = np.column_stack((points_real, points_imag))
        return samples

    # Mandelbrot set convergence check
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

    # Calculate the area of the Mandelbrot set
    def calcu_mandelbrot_area(self, samples, max_iter, plane_area = 16):
        mask = self.mandel_convergence_check_vectorized(samples, max_iter)
        area_ratio = np.sum(mask) / len(mask)
        area = area_ratio * plane_area
        area = round (area, 6)
        return area

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
        plt.title(f'Visualization of the Mandelbrot Set ({sample_name} Random Sampling with {len(samples)} Samples and {max_iter} Iterations)')
        plt.legend()
        
        # store the image into a file, if no existing directory, create one        
        os.makedirs(IMG_COLOR_DIR, exist_ok=True)
        plt.savefig(f'{IMG_COLOR_DIR}/mandelbrot_{sample_name}_{len(samples)}_maxIter_{max_iter}.png')

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
    
    #--------------------------------------------------------------improvement the convergence----------------------------------------------
    def divide_complex_plane(self, dimension_separate_number):
        real_parts = np.linspace(self.real_range[0], self.real_range[1], dimension_separate_number + 1)
        imag_parts = np.linspace(self.imag_range[0], self.imag_range[1], dimension_separate_number + 1)
        regions = []
        for i in range(dimension_separate_number):
            for j in range(dimension_separate_number):
                regions.append(((real_parts[i], real_parts[i+1]), (imag_parts[j], imag_parts[j+1])))
        return regions
    
    def complexity_measure(self, region):
        num_repeats = 5
        areas = []
        for _ in range(num_repeats):
            real_range, imag_range = region
            num_samples_root = 100
            # use random to check the variance of the area
            region_samples = self.pure_random_sampling_partial(num_samples_root, real_range[0], real_range[1], imag_range[0], imag_range[1])
            area = self.calcu_mandelbrot_area(region_samples, 100)
            areas.append(area)
        # return the variance of the areas
        epsilon = 1e-10
        return np.var(areas) + epsilon

    def adaptive_sampling(self, num_samples_root, dimension_separate_number):
        total_samples_numbers = num_samples_root * num_samples_root
        regions = self.divide_complex_plane(dimension_separate_number)
        region_complexities = []
        real_and_imag_parts = []

        for region in regions:
            real_range, imag_range = region
            complexity = self.complexity_measure(region)
            region_complexities.append(complexity)
            real_and_imag_parts.append((real_range, imag_range))

        # calculate the weight of each region based on the complexity
        total_complexity = np.sum(region_complexities)
        weights = region_complexities / total_complexity

        # according the weight to separate the num_samples_root into different regions
        # smoothly separate the samples number, for the biggest weights, we just give half of the separated samples,
        # then, for the rest, we separate the samples based on the weights, but still keep the integer number of samples.
        separate_samples_root = np.zeros(len(weights), dtype=int)
        max_weight_index = np.argmax(weights)
        separate_samples_root[max_weight_index] = int(round(np.sqrt(total_samples_numbers // 4)))

        for i in range(len(weights)):
            if i != max_weight_index:
                separate_samples_root[i] = int(round(np.sqrt(int((total_samples_numbers - total_samples_numbers // 4) * weights[i] / (1 - weights[max_weight_index])))))
                
        # now we have region_complexities, real_and_imag_parts, weights and separate_samples_root with the same index.
        total_samples = []
        for i in range(len(region_complexities)):
            refined_samples = self.orthogonal_sampling_partial(separate_samples_root[i], real_and_imag_parts[i][0][0], real_and_imag_parts[i][0][1], real_and_imag_parts[i][1][0], real_and_imag_parts[i][1][1])
            #print(f"Region {i} has {len(refined_samples)} samples.")
            total_samples.append(refined_samples)

        # np.concatenate(total_samples)
        return total_samples

if __name__ == "__main__":
    mandelbrot = MandelbrotAnalysis(real_range=(-2, 1), imag_range=(-1.5, 1.5))
    num_samples = 1000
    min_iter = 10
    max_iter = 100
    mandelbrot.compare_sampling_methods(num_samples, min_iter, max_iter)
    
    
