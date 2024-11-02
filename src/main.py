import multiprocessing as mp
from joblib import Parallel, delayed
import itertools
import mandelbrot_analysis

# initialize the MandelbrotAnalysis platform
mandelbrotAnalysisPlatform = mandelbrot_analysis.MandelbrotAnalysis(real_range=(-2, 2), imag_range=(-2, 2))

# -----------------------------------------------------------color_mandelbrot-----------------------------------------------------------
# pick the best combination of num_samples and max_iter
num_samples_list = [1000, 5000, 10000]
max_iter_list = [100, 200]
mset_list = list(itertools.product(num_samples_list, max_iter_list))

def run_mset_colors(num_samples, max_iter):
    # 0 is for pure random sampling
    # TODO: Implement pure random sampling

    # 1 is for LHS sampling
    sample = mandelbrotAnalysisPlatform.latin_hypercube_sampling(num_samples)
    mandelbrotAnalysisPlatform.color_mandelbrot(sample, max_iter, 1)

    # 2 is for orthogonal sampling
    # TODO: Implement orthogonal sampling

num_workers = mp.cpu_count()
results = Parallel(n_jobs=num_workers)(delayed(run_mset_colors)(num_samples, max_iter) for num_samples, max_iter in mset_list)

# -----------------------------------------------------------inverstigate convergence-----------------------------------------------------------
