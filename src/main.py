import os
import sys
import threading
import time
import multiprocessing as mp
from joblib import Parallel, delayed
import itertools
import mandelbrot_analysis
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

stop_event = threading.Event()
def show_wait_message(msg = "Hang in there, it's almost done"):
    animation = ["", ".", "..", "..."]
    idx = 0
    while not stop_event.is_set():
        print(" " * 100, end="\r")
        print(f"{msg}{animation[idx % len(animation)]}", end="\r")
        idx += 1
        time.sleep(0.5)

# initialize the MandelbrotAnalysis platform
mandelbrotAnalysisPlatform = mandelbrot_analysis.MandelbrotAnalysis(real_range=(-2, 2), imag_range=(-2, 2))

# -----------------------------------------------------------color_mandelbrot-----------------------------------------------------------
def mset_colors_parallel(num_samples, max_iter):
    # 0 is for pure random sampling
    sample = mandelbrotAnalysisPlatform.pure_random_sampling(num_samples)
    mandelbrotAnalysisPlatform.color_mandelbrot(sample, max_iter, 0)

    # 1 is for LHS sampling
    sample = mandelbrotAnalysisPlatform.latin_hypercube_sampling(num_samples)
    mandelbrotAnalysisPlatform.color_mandelbrot(sample, max_iter, 1)

def mset_colors_ortho_seq(num_samples_list_perfect_root, max_iter_list):
    # 2 corresponds to orthogonal sampling
    # joblib is not able to seriliaze the object which has ctypes pointer, so we have to run this sequentially
    for i, num_samples in enumerate(num_samples_list_perfect_root):
        for j, max_iter in enumerate(max_iter_list):
            sample = mandelbrotAnalysisPlatform.orthogonal_sampling(num_samples)
            mandelbrotAnalysisPlatform.color_mandelbrot(sample, max_iter, 2)
            
def run_mset_colors():
    # pick the best combination of num_samples and max_iter
    num_samples_list              = [6400, 10000, 90000] # 50, 80, 100
    num_samples_list_perfect_root = [80, 100, 300] # perfect square root for orthogonal sampling, sample size is square of this number!!!
    max_iter_list = [100, 200]
    mset_list = list(itertools.product(num_samples_list, max_iter_list))

    # pure random sampling and LHS sampling can be run in parallel
    num_workers = mp.cpu_count()
    results = Parallel(n_jobs=num_workers)(delayed(mset_colors_parallel)(num_samples, max_iter) for num_samples, max_iter in mset_list)

    # orthogonal sampling has to be run sequentially
    mandelbrotAnalysisPlatform._load_library()
    mset_colors_ortho_seq(num_samples_list_perfect_root, max_iter_list)


# -----------------------------------------------------------inverstigate convergence-----------------------------------------------------------
def get_and_save_true_area():
    max_num_samples_root = 300
    max_iter = 300
    sample = mandelbrotAnalysisPlatform.orthogonal_sampling(max_num_samples_root)
    area = mandelbrotAnalysisPlatform.calcu_mandelbrot_area(sample, max_iter)
    print(f"Ture Area of the Mandelbrot set samples is {area}")
    # Save the result to a file
    with open("tureArea.txt", "w") as file:
        file.write(f"True Area of the Mandelbrot set samples is {area:.6f}\n")
    
    return area

def read_area_from_file():
    try:
        # Open the file and read the area value
        with open("tureArea.txt", "r") as file:
            line = file.readline()
            value = line.split()[-1]
            if value.replace('.', '', 1).isdigit():
                alpha = float(value)
            else:
                alpha = 0
    except (FileNotFoundError, ValueError):
        alpha = 0
    return alpha

def save_area_series_into_files():
    # pick the best combination of num_samples and max_iter
    num_samples_list_perfect_root = [80, 100, 120, 140, 160, 200]
    max_iter_list = [50, 100, 150, 180, 200, 220, 230]
    mset_list = list(itertools.product(num_samples_list_perfect_root, max_iter_list))

    for sample_type in [0, 1, 2]:
        sample_name = mandelbrotAnalysisPlatform.get_sample_name(sample_type)
        num_samples_vals, max_iter_vals, area_vals = get_mset_area_collection(mset_list, sample_type)
        # Save pure random sampling data to file
        with open(f"mandelbrotArea_{sample_name}.txt", "w") as file:
            for num_samples, max_iter, area in zip(num_samples_vals, max_iter_vals, area_vals):
                file.write(f"{num_samples} {max_iter} {area:.6f}\n")

def read_area_series_from_files():
    area_data = {}
    for sample_type in [0, 1, 2]:
        sample_name = mandelbrotAnalysisPlatform.get_sample_name(sample_type)
        try:
            with open(f"mandelbrotArea_{sample_name}.txt", "r") as file:
                area_data[sample_name] = []
                for line in file:
                    num_samples, max_iter, area = line.split()
                    area_data[sample_name].append((int(num_samples), int(max_iter), float(area)))
        except FileNotFoundError:
            print(f"File mandelbrotArea_{sample_name}.txt not found.")
            area_data[sample_name] = []
    return area_data

def get_mset_area_collection(mset_list, sample_type=0):
    # read the true area from the file
    alpha = read_area_from_file()
    if alpha == 0:
        alpha = get_and_save_true_area()
    
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
        if abs(area - alpha) < 0.001:
            print(f"Convergence reached with method {sample_name}, {num_samples} samples and {max_iter} max iterations")

    return num_samples_vals, max_iter_vals, area_vals

def run_mset_statistic_and_plot():
    if mandelbrotAnalysisPlatform.lib is None:
        mandelbrotAnalysisPlatform._load_library()

    # Try to read data from files
    alpha = read_area_from_file()
    if alpha == 0:
        alpha = get_and_save_true_area()
    
    area_data_set = read_area_series_from_files()

    # Check if data exists for all sampling methods, if not, generate and save it
    if not all(area_data_set[mandelbrotAnalysisPlatform.get_sample_name(sample_type)] for sample_type in [0, 1, 2]):
        save_area_series_into_files()
        area_data_set = read_area_series_from_files()

    # Extract data for plotting
    num_samples_vals1, max_iter_vals1, area_vals1 = zip(*area_data_set["Pure"]) if area_data_set["Pure"] else ([], [], [])
    num_samples_vals2, max_iter_vals2, area_vals2 = zip(*area_data_set["LHS"]) if area_data_set["LHS"] else ([], [], [])
    num_samples_vals3, max_iter_vals3, area_vals3 = zip(*area_data_set["Ortho"]) if area_data_set["Ortho"] else ([], [], [])

    # Calculate differences from alpha
    area_diff_vals1 = [area - alpha for area in area_vals1]
    area_diff_vals2 = [area - alpha for area in area_vals2]
    area_diff_vals3 = [area - alpha for area in area_vals3]

    # Plot the results in a 3D plot
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(num_samples_vals1, max_iter_vals1, area_diff_vals1, c='b', marker='o', label='Pure Random Sampling')
    ax.scatter(num_samples_vals2, max_iter_vals2, area_diff_vals2, c='r', marker='^', label='LHS Sampling')
    ax.scatter(num_samples_vals3, max_iter_vals3, area_diff_vals3, c='g', marker='s', label='Orthogonal Sampling')
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Max Iterations')
    ax.set_zlabel('Area Difference (Area - Alpha)')
    ax.set_title('Mandelbrot Set Area Analysis')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    # Create a surface plot to connect the points with different colors and add edge lines
    ax.plot_trisurf(num_samples_vals1, max_iter_vals1, area_diff_vals1, color='b', alpha=0.5, edgecolor='k', linewidth=0.5)
    ax.plot_trisurf(num_samples_vals2, max_iter_vals2, area_diff_vals2, color='r', alpha=0.5, edgecolor='k', linewidth=0.5)
    ax.plot_trisurf(num_samples_vals3, max_iter_vals3, area_diff_vals3, color='g', alpha=0.5, edgecolor='k', linewidth=0.5)


    # store the image into a file, if no existing directory, create one        
    output_dir = '../images/convergence_analysis'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/converge_3D.png')
    
# -----------------------------------------------------------sampling statistical analysis-----------------------------------------------------
def run_sampling_statistic_analysis():
    time.sleep(20)

# -----------------------------------------------------------main controller process-----------------------------------------------------------
def main_controller():
    while True:
        print("*" * 80)
        print("Select an option to run:")
        print("1: Run Mandelbrot color plottings")
        print("2: Run Mandelbrot convergence analysis")
        print("3: Run Mandelbrot sampling statistical analysis")
        print("0: Exit")
        
        try:
            choice = int(input("Enter your choice: "))
        except ValueError:
            print("Invalid input, please enter a number.")
            continue

        if choice == 1:
            wait_thread = threading.Thread(target=show_wait_message, args=("Running Mandelbrot color plottings, please wait ",))
            wait_thread.start()
            run_mset_colors()
            stop_event.set()
            wait_thread.join()

        elif choice == 2:
            wait_thread = threading.Thread(target=show_wait_message, args=("Running Mandelbrot convergence analysis, please wait ",))
            wait_thread.start()
            run_mset_statistic_and_plot()
            stop_event.set()
            wait_thread.join()

        elif choice == 3:
            wait_thread = threading.Thread(target=show_wait_message, args=("Running Mandelbrot sampling statistic analysis, please wait ",))
            wait_thread.start()
            run_sampling_statistic_analysis()
            stop_event.set()
            wait_thread.join()

        elif choice == 0:
            print("Exiting the program.")
            break
        else:
            print("Invalid choice, please select a valid option.")

if __name__ == "__main__":
    main_controller()

