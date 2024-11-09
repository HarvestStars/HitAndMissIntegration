import itertools
import os

RESULT_DIR = '../simulation_results'
STATISTIC_RESULT_DIR = '../simulation_results/same_iter_and_size'
CONVERGENCE_ANALYSIS_RESULT_DIR = '../simulation_results/same_size_diff_iter'

# -----------------------------------------------------------color_mandelbrot-----------------------------------------------------------
def mset_colors_parallel(mandelbrotAnalysisPlatform, num_samples, max_iter):
    # 0 is for pure random sampling
    sample = mandelbrotAnalysisPlatform.pure_random_sampling(num_samples)
    mandelbrotAnalysisPlatform.color_mandelbrot(sample, max_iter, 0)

    # 1 is for LHS sampling
    sample = mandelbrotAnalysisPlatform.latin_hypercube_sampling(num_samples)
    mandelbrotAnalysisPlatform.color_mandelbrot(sample, max_iter, 1)

def mset_colors_ortho_seq(mandelbrotAnalysisPlatform, num_samples_list_perfect_root, max_iter_list):
    # 2 corresponds to orthogonal sampling
    # joblib is not able to seriliaze the object which has ctypes pointer, so we have to run this sequentially
    for i, num_samples in enumerate(num_samples_list_perfect_root):
        for j, max_iter in enumerate(max_iter_list):
            sample = mandelbrotAnalysisPlatform.orthogonal_sampling(num_samples)
            mandelbrotAnalysisPlatform.color_mandelbrot(sample, max_iter, 2)
        
# -----------------------------------------------------------inverstigate convergence-----------------------------------------------------------
def get_and_save_true_area(mandelbrotAnalysisPlatform):
    max_num_samples_root = 400
    max_iter = 300
    sample = mandelbrotAnalysisPlatform.orthogonal_sampling(max_num_samples_root)
    area = mandelbrotAnalysisPlatform.calcu_mandelbrot_area(sample, max_iter)
    print(f"True Area of the Mandelbrot set samples is {area}")
    # Save the result to a file
    with open(f'{RESULT_DIR}/trueArea.txt', "w") as file:
        file.write(f"True Area of the Mandelbrot set samples is {area:.6f}\n")
    
    return area

def read_area_from_file():
    try:
        # Open the file and read the area value
        with open(f'{RESULT_DIR}/trueArea.txt', "r") as file:
            line = file.readline()
            value = line.split()[-1]
            if value.replace('.', '', 1).isdigit():
                alpha = float(value)
            else:
                alpha = 0
    except (FileNotFoundError, ValueError):
        alpha = 0
    return alpha

def save_area_series_into_files(mandelbrotAnalysisPlatform):
    # pick the best combination of num_samples and max_iter
    num_samples_list_perfect_root = [80, 100, 120, 140, 160, 200]
    max_iter_list = [50, 100, 150, 180, 200, 220, 230]
    mset_list = list(itertools.product(num_samples_list_perfect_root, max_iter_list))

    for sample_type in [0, 1, 2]:
        sample_name = mandelbrotAnalysisPlatform.get_sample_name(sample_type)
        num_samples_vals, max_iter_vals, area_vals = get_mset_area_collection(mandelbrotAnalysisPlatform, mset_list, sample_type)
        # Save pure random sampling data to file
        with open(f'{RESULT_DIR}/mandelbrotArea_{sample_name}.txt', "w") as file:
            for num_samples, max_iter, area in zip(num_samples_vals, max_iter_vals, area_vals):
                file.write(f"{num_samples} {max_iter} {area:.6f}\n")

def save_area_series_into_files_with_fix_iter_and_size(mandelbrotAnalysisPlatform):
    repeat = 100
    mset_list = [(400, 300) for _ in range(repeat)]

    for sample_type in [0, 1, 2]:
        sample_name = mandelbrotAnalysisPlatform.get_sample_name(sample_type)
        num_samples_vals, max_iter_vals, area_vals = get_mset_area_collection(mandelbrotAnalysisPlatform, mset_list, sample_type)
        # Save pure random sampling data to file
        os.makedirs(STATISTIC_RESULT_DIR, exist_ok=True)
        with open(f'{STATISTIC_RESULT_DIR}/mandelbrotArea_{sample_name}.txt', "w") as file:
            for num_samples, max_iter, area in zip(num_samples_vals, max_iter_vals, area_vals):
                file.write(f"{num_samples} {max_iter} {area:.6f}\n")

def save_area_series_into_files_with_fix_size_but_differ_iters(mandelbrotAnalysisPlatform, sample_size, max_iter):
    mset_list = [(sample_size, i) for i in range(60, max_iter, 20)]

    for sample_type in [0, 1, 2]:
        sample_name = mandelbrotAnalysisPlatform.get_sample_name(sample_type)
        num_samples_vals, max_iter_vals, area_vals = get_mset_area_collection(mandelbrotAnalysisPlatform, mset_list, sample_type)
        # Save pure random sampling data to file
        os.makedirs(CONVERGENCE_ANALYSIS_RESULT_DIR, exist_ok=True)
        with open(f'{CONVERGENCE_ANALYSIS_RESULT_DIR}/mandelbrotArea_{sample_name}.txt', "w") as file:
            for num_samples, max_iter, area in zip(num_samples_vals, max_iter_vals, area_vals):
                file.write(f"{num_samples} {max_iter} {area:.6f}\n")

def read_area_series_from_files(mandelbrotAnalysisPlatform):
    area_data = {}
    for sample_type in [0, 1, 2]:
        sample_name = mandelbrotAnalysisPlatform.get_sample_name(sample_type)
        try:
            with open(f'{RESULT_DIR}/mandelbrotArea_{sample_name}.txt', "r") as file:
                area_data[sample_name] = []
                for line in file:
                    num_samples, max_iter, area = line.split()
                    area_data[sample_name].append((int(num_samples), int(max_iter), float(area)))
        except FileNotFoundError:
            print(f"File {RESULT_DIR}/mandelbrotArea_{sample_name}.txt not found.")
            area_data[sample_name] = []
    return area_data

def read_area_series_from_files_with_fix_size_but_differ_iters(mandelbrotAnalysisPlatform):
    area_data = {}
    for sample_type in [0, 1, 2]:
        sample_name = mandelbrotAnalysisPlatform.get_sample_name(sample_type)
        try:
            with open(f'{CONVERGENCE_ANALYSIS_RESULT_DIR}/mandelbrotArea_{sample_name}.txt', "r") as file:
                area_data[sample_name] = []
                for line in file:
                    num_samples, max_iter, area = line.split()
                    area_data[sample_name].append((int(num_samples), int(max_iter), float(area)))
        except FileNotFoundError:
            print(f"File {CONVERGENCE_ANALYSIS_RESULT_DIR}/mandelbrotArea_{sample_name}.txt not found.")
            area_data[sample_name] = []
    return area_data

def get_mset_area_collection(mandelbrotAnalysisPlatform, mset_list, sample_type=0):
    # read the true area from the file
    alpha = read_area_from_file()
    if alpha == 0:
        alpha = get_and_save_true_area(mandelbrotAnalysisPlatform)
    
    # Initialize lists to store data for 3D plotting
    num_samples_vals = []
    max_iter_vals = []
    area_vals = []
    sample_name = mandelbrotAnalysisPlatform.get_sample_name(sample_type)

    # run the area collection
    for num_samples_root, max_iter in mset_list:
        num_samples = num_samples_root**2
        if sample_name == "Pure":
            sample = mandelbrotAnalysisPlatform.pure_random_sampling(num_samples)
        elif sample_name == "LHS":
            sample = mandelbrotAnalysisPlatform.latin_hypercube_sampling(num_samples)
        elif sample_name == "Ortho":
            sample = mandelbrotAnalysisPlatform.orthogonal_sampling(num_samples_root)
        else:
            sample = mandelbrotAnalysisPlatform.pure_random_sampling(num_samples)

        area = mandelbrotAnalysisPlatform.calcu_mandelbrot_area(sample, max_iter)
        print(f"Area of the Mandelbrot set with method {sample_name}, {num_samples} samples and {max_iter} max iterations is {area}")

        # Store data for 3D plotting
        num_samples_vals.append(num_samples)
        max_iter_vals.append(max_iter)
        area_vals.append(area)

        # check the convergence
        if abs(area - alpha) < 0.00001:
            print(f"Convergence reached with method {sample_name}, {num_samples} samples and {max_iter} max iterations")

    return num_samples_vals, max_iter_vals, area_vals