import itertools
import os
import matplotlib.pyplot as plt
import numpy as np

RESULT_DIR = '../simulation_results'
STATISTIC_RESULT_DIR = '../simulation_results/same_iter_and_size'

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
    max_num_samples_root = 3000
    max_iter = 1000
    sample = mandelbrotAnalysisPlatform.orthogonal_sampling(max_num_samples_root)
    plane_area = abs(mandelbrotAnalysisPlatform.real_range[1] - mandelbrotAnalysisPlatform.real_range[0]) * (mandelbrotAnalysisPlatform.imag_range[1] - mandelbrotAnalysisPlatform.imag_range[0])
    area = mandelbrotAnalysisPlatform.calcu_mandelbrot_area(sample, max_iter, plane_area)
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
    num_samples_list_perfect_root = [500, 800, 1000, 1600, 2000, 2400, 2600, 3000]
    max_iter_list = [100, 150, 200, 240, 300, 400, 600, 700, 800, 900, 1000]
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
    mset_list = [(2600, 800) for _ in range(repeat)]

    for sample_type in [0, 1, 2]:
        sample_name = mandelbrotAnalysisPlatform.get_sample_name(sample_type)
        num_samples_vals, max_iter_vals, area_vals = get_mset_area_collection(mandelbrotAnalysisPlatform, mset_list, sample_type)
        # Save pure random sampling data to file
        os.makedirs(STATISTIC_RESULT_DIR, exist_ok=True)
        with open(f'{STATISTIC_RESULT_DIR}/mandelbrotArea_{sample_name}.txt', "w") as file:
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

        plane_area = abs(mandelbrotAnalysisPlatform.real_range[1] - mandelbrotAnalysisPlatform.real_range[0]) * (mandelbrotAnalysisPlatform.imag_range[1] - mandelbrotAnalysisPlatform.imag_range[0])
        area = mandelbrotAnalysisPlatform.calcu_mandelbrot_area(sample, max_iter, plane_area)
        print(f"Area of the Mandelbrot set with method {sample_name}, {num_samples} samples and {max_iter} max iterations is {area}")

        # Store data for 3D plotting
        num_samples_vals.append(num_samples)
        max_iter_vals.append(max_iter)
        area_vals.append(area)

        # check the convergence
        if abs(area - alpha) < 0.00001:
            print(f"Convergence reached with method {sample_name}, {num_samples} samples and {max_iter} max iterations")

    return num_samples_vals, max_iter_vals, area_vals

# Plot individual 3D plots for each sampling method
def plot_individual_3d(num_samples_vals, max_iter_vals, area_diff_vals, color, marker, label, filename):
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(num_samples_vals, max_iter_vals, area_diff_vals, c=color, marker=marker, label=label)
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Max Iterations')
    ax.set_zlabel('Madbelbrot Set Area with different i and s')
    ax.set_title(f'Mandelbrot Set Area Analysis - {label}')
    ax.legend()
    ax.invert_yaxis()  # Invert the y-axis to make max_iter_vals appear in descending order from near to far
    ax.plot_trisurf(num_samples_vals, max_iter_vals, area_diff_vals, color=color, alpha=0.5, edgecolor='k', linewidth=0.5)
    plt.savefig(filename)
    plt.close()

# Generate heatmaps for each of the datasets separately
def generate_heatmap(data_x, data_y, data_z, title, xlabel, ylabel, filename):
    # Create a grid for plotting heatmap values
    x_edges = np.unique(data_x)  # x_edges to represent iterations (max_iter_vals)
    y_edges = np.unique(data_y)  # y_edges to represent number of samples (num_samples_vals)
    X, Y = np.meshgrid(x_edges, y_edges)
    Z = np.zeros_like(X, dtype=float)

    # Populate the Z values based on input data
    for i in range(len(data_x)):
        x_idx = np.where(x_edges == data_x[i])[0][0]
        y_idx = np.where(y_edges == data_y[i])[0][0]
        Z[y_idx, x_idx] = data_z[i]

    # Plotting the heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    c = ax.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
    ax.set_xticks(x_edges)
    ax.set_yticks(y_edges)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.colorbar(c, ax=ax)
    plt.savefig(filename)
    plt.close()

def plot_convergence_curve(num_samples_vals, max_iter_vals, area_vals_diff, method_name, filename_prefix):
    # Plot convergence with respect to iterations for different sample sizes
    fig, ax = plt.subplots(figsize=(10, 8))
    for sample_size in np.unique(num_samples_vals):
        iter_vals = [max_iter_vals[i] for i in range(len(num_samples_vals)) if num_samples_vals[i] == sample_size]
        #area_diffs = [0 if i == 0 or num_samples_vals[i] != num_samples_vals[i - 1] else abs(area_vals[i] - area_vals[i - 1]) for i in range(len(num_samples_vals)) if num_samples_vals[i] == sample_size]
        area_diff_fix_s = [abs(area_vals_diff[i]) for i in range(len(num_samples_vals)) if num_samples_vals[i] == sample_size]
        ax.plot(iter_vals, area_diff_fix_s, label=f'Sample Size = {sample_size}')

    ax.set_xlabel('Iterations')
    ax.set_ylabel('Area Difference (Current - True Area)')
    ax.set_title(f'Convergence Analysis - {method_name} (vs Iterations)')
    ax.legend()
    ax.grid()
    plt.savefig(f'{filename_prefix}_iterations.png')
    plt.close()

    # Plot convergence with respect to sample sizes for different iteration limits
    fig, ax = plt.subplots(figsize=(10, 8))
    for iter_limit in np.unique(max_iter_vals):
        sample_vals = [num_samples_vals[i] for i in range(len(max_iter_vals)) if max_iter_vals[i] == iter_limit]
        # area_diffs = [0 if i < len(np.unique(max_iter_vals)) else abs(area_vals[i] - area_vals[i - len(np.unique(max_iter_vals))]) for i in range(len(max_iter_vals)) if max_iter_vals[i] == iter_limit]
        area_diffs_fix_i = [abs(area_vals_diff[i]) for i in range(len(max_iter_vals)) if max_iter_vals[i] == iter_limit]
        ax.plot(sample_vals, area_diffs_fix_i, label=f'Iteration Limit = {iter_limit}')

    ax.set_xlabel('Sample Sizes')
    ax.set_ylabel('Area Difference (Current - True Area)')
    ax.set_title(f'Convergence Analysis - {method_name} (vs Sample Sizes)')
    ax.legend()
    ax.grid()
    plt.savefig(f'{filename_prefix}_samples.png')
    plt.close()

def plot_convergence_comparison(area_data_set, trueArea, filename_prefix):
    # Select a fixed sample size (the first sample size in the data set)
    # fixed_sample_size = list(area_data_set.values())[0][0][0]
    fixed_sample_size = 2560000

    plt.figure(figsize=(10, 6))
    line_styles = [('-', 'o'), ('--', 's'), (':', '^'), ('-.', 'd')]

    # Plot convergence curves for each method
    for idx, (method_name, area_data) in enumerate(area_data_set.items()):
        # Filter data for the fixed sample size
        filtered_data = [data for data in area_data if data[0] == fixed_sample_size]
        if filtered_data:
            max_iter_vals = [data[1] for data in filtered_data]
            area_vals = [abs(data[2] - trueArea) for data in filtered_data]
            linestyle, marker = line_styles[idx % len(line_styles)]
            plt.plot(max_iter_vals, area_vals, label=method_name, linewidth=2, linestyle=linestyle, marker=marker)

    # Plot true area as a reference line
    plt.axhline(y=0, color='gray', linestyle='--', label='diff = 0')
    plt.xlabel('Iterations')
    plt.ylabel('Area Value')
    plt.title(f'Convergence Comparison for Sample Size {fixed_sample_size}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{filename_prefix}_convergence_comparison.png")
    plt.close()
