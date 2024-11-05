import multiprocessing as mp
from joblib import Parallel, delayed
import itertools
import mandelbrot_analysis

# initialize the MandelbrotAnalysis platform
mandelbrotAnalysisPlatform = mandelbrot_analysis.MandelbrotAnalysis(real_range=(-2, 2), imag_range=(-2, 2))

# -----------------------------------------------------------color_mandelbrot-----------------------------------------------------------
# pick the best combination of num_samples and max_iter
num_samples_list              = [2500, 6400, 10000] # 50, 80, 100
num_samples_list_perfect_root = [50, 80, 100] # perfect square root for orthogonal sampling, sample size is square of this number!!!
max_iter_list = [100, 200]
mset_list = list(itertools.product(num_samples_list, max_iter_list))

def run_mset_colors_parallel(num_samples, max_iter):
    # 0 is for pure random sampling
    # TODO: Implement pure random sampling

    # 1 is for LHS sampling
    sample = mandelbrotAnalysisPlatform.latin_hypercube_sampling(num_samples)
    mandelbrotAnalysisPlatform.color_mandelbrot(sample, max_iter, 1)

def run_mset_colors_ortho_seq(num_samples_list_perfect_root, max_iter_list):
    # 2 corresponds to orthogonal sampling
    # joblib is not able to seriliaze the object which has ctypes pointer, so we have to run this sequentially
    for i, num_samples in enumerate(num_samples_list_perfect_root):
        for j, max_iter in enumerate(max_iter_list):
            sample = mandelbrotAnalysisPlatform.orthogonal_sampling(num_samples)
            mandelbrotAnalysisPlatform.color_mandelbrot(sample, max_iter, 2)
            
def run_mset_colors_pure_seq(num_samples_list, max_iter_list):
    """
    This function applys purely random sampling to approximate a Mandelbrot set, with given
    number of samples and times of iteration
    """
    for i, num_samples in enumerate(num_samples_list):
        for j, max_iter in enumerate(max_iter_list):
            sample = mandelbrotAnalysisPlatform.pure_random_sampling(num_samples)
            mandelbrotAnalysisPlatform.color_mandelbrot(sample, max_iter, 0)

num_workers = mp.cpu_count()
results = Parallel(n_jobs=num_workers)(delayed(run_mset_colors_parallel)(num_samples, max_iter) for num_samples, max_iter in mset_list)

mandelbrotAnalysisPlatform._load_library()
run_mset_colors_ortho_seq(num_samples_list_perfect_root, max_iter_list)
run_mset_colors_pure_seq(num_samples_list, max_iter_list)

# -----------------------------------------------------------inverstigate convergence-----------------------------------------------------------
