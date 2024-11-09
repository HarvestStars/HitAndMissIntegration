import os
import sys
import threading
import time
import multiprocessing as mp
from joblib import Parallel, delayed
import itertools
import mandelbrot_analysis
import utils
import metrics
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
def run_mset_colors():
    # pick the best combination of num_samples and max_iter
    num_samples_list              = [6400, 10000, 90000] # 50, 80, 100
    num_samples_list_perfect_root = [80, 100, 300] # perfect square root for orthogonal sampling, sample size is square of this number!!!
    max_iter_list = [100, 200]
    mset_list = list(itertools.product(num_samples_list, max_iter_list))

    # pure random sampling and LHS sampling can be run in parallel
    num_workers = mp.cpu_count()
    results = Parallel(n_jobs=num_workers)(delayed(utils.mset_colors_parallel)(mandelbrotAnalysisPlatform, num_samples, max_iter) for num_samples, max_iter in mset_list)

    # orthogonal sampling has to be run sequentially
    mandelbrotAnalysisPlatform._load_library()
    utils.mset_colors_ortho_seq(num_samples_list_perfect_root, max_iter_list)


# -----------------------------------------------------------generate true area-----------------------------------------------------------------
def run_generate_true_area():
    if mandelbrotAnalysisPlatform.lib is None:
        mandelbrotAnalysisPlatform._load_library()
    utils.get_and_save_true_area(mandelbrotAnalysisPlatform)


# -----------------------------------------------------------inverstigate convergence-----------------------------------------------------------
def run_mset_statistic_and_plot():
    if mandelbrotAnalysisPlatform.lib is None:
        mandelbrotAnalysisPlatform._load_library()

    # Try to read data from files
    alpha = utils.read_area_from_file()
    if alpha == 0:
        alpha = utils.get_and_save_true_area(mandelbrotAnalysisPlatform)
    
    area_data_set = utils.read_area_series_from_files(mandelbrotAnalysisPlatform)

    # Check if data exists for all sampling methods, if not, generate and save it
    if not all(area_data_set[mandelbrotAnalysisPlatform.get_sample_name(sample_type)] for sample_type in [0, 1, 2]):
        utils.save_area_series_into_files(mandelbrotAnalysisPlatform)
        area_data_set = utils.read_area_series_from_files(mandelbrotAnalysisPlatform)

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
    os.makedirs(mandelbrot_analysis.IMG_CONVERGENCE_DIR, exist_ok=True)
    plt.savefig(f'{mandelbrot_analysis.IMG_CONVERGENCE_DIR}/converge_3D.png')
    plt.close()


# -----------------------------------------------------------inverstigate convergence for fixed sample size------------------------------------
def run_mset_statistic_and_plot_fixed_sample_size():
    if mandelbrotAnalysisPlatform.lib is None:
        mandelbrotAnalysisPlatform._load_library()

    # Try to read data from files
    alpha = utils.read_area_from_file()
    if alpha == 0:
        alpha = utils.get_and_save_true_area(mandelbrotAnalysisPlatform)

    area_data_set = utils.read_area_series_from_files_with_fix_size_but_differ_iters(mandelbrotAnalysisPlatform)

    # Check if data exists for all sampling methods, if not, generate and save it
    if not all(area_data_set[mandelbrotAnalysisPlatform.get_sample_name(sample_type)] for sample_type in [0, 1, 2]):
        print(f"No data found for fixed sample size and varying iterations, generating and saving data...")
        utils.save_area_series_into_files_with_fix_size_but_differ_iters(mandelbrotAnalysisPlatform, 400, 500)
        area_data_set = utils.read_area_series_from_files_with_fix_size_but_differ_iters(mandelbrotAnalysisPlatform)

    # Extract data for plotting
    num_samples_vals1, max_iter_vals1, area_vals1 = zip(*area_data_set["Pure"]) if area_data_set["Pure"] else ([], [], [])
    num_samples_vals2, max_iter_vals2, area_vals2 = zip(*area_data_set["LHS"]) if area_data_set["LHS"] else ([], [], [])
    num_samples_vals3, max_iter_vals3, area_vals3 = zip(*area_data_set["Ortho"]) if area_data_set["Ortho"] else ([], [], [])

    # Calculate differences from alpha
    area_diff_vals1 = [area - alpha for area in area_vals1]
    area_diff_vals2 = [area - alpha for area in area_vals2]
    area_diff_vals3 = [area - alpha for area in area_vals3]
    
    # Plot the area differences as 2D lines against max iterations
    plt.figure(figsize=(12, 8))
    plt.plot(max_iter_vals1, area_diff_vals1, label='Pure Random Sampling', color='b', linestyle='-', linewidth=2, marker='o', markersize=6)
    plt.plot(max_iter_vals2, area_diff_vals2, label='LHS Sampling', color='r', linestyle='--', linewidth=2, marker='^', markersize=6)
    plt.plot(max_iter_vals3, area_diff_vals3, label='Orthogonal Sampling', color='g', linestyle='-.', linewidth=2, marker='s', markersize=6)
    plt.xlabel('Max Iterations', fontsize=14)
    plt.ylabel('Area Difference (Area - Alpha)', fontsize=14)
    plt.title('Area Difference vs Max Iterations', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Store the 2D plot into a file
    plt.savefig(f'{mandelbrot_analysis.IMG_CONVERGENCE_DIR}/area_diff_vs_iterations.png')
    plt.close()



# -----------------------------------------------------------statistic sample generate---------------------------------------------------------
def run_statistic_sample_generate():
    if mandelbrotAnalysisPlatform.lib is None:
        mandelbrotAnalysisPlatform._load_library()
    utils.save_area_series_into_files_with_fix_iter_and_size(mandelbrotAnalysisPlatform)
    

# -----------------------------------------------------------statistic metrics-----------------------------------------------------------------
def run_statistic_metric():
    mean_and_variance = metrics.calculate_mean_and_variance()
    print("Mean and Variance:", mean_and_variance)

    mse = metrics.calculate_mse()
    print("Mean Squared Error (MSE):", mse)

    confidence_intervals = metrics.calculate_confidence_intervals()
    print("Confidence Intervals:", confidence_intervals)

    # Plot area distributions
    metrics.plot_area_distributions()

# -----------------------------------------------------------main controller process-----------------------------------------------------------
def main_controller():
    while True:
        print("*" * 80)
        print("Select an option to run:")
        print("1: Run Mandelbrot color plottings")
        print("2: Run Generate True Area")
        print("3: Run Mandelbrot convergence analysis for different parameters")
        print("4: Run Mandelbrot convergence analysis for fixed sample size and varying iterations")
        print("5: Run Mandelbrot statistic sample generate")
        print("6: Run Mandelbrot statistic metrics and plots")
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
            wait_thread = threading.Thread(target=show_wait_message, args=("Running Mandelbrot color plottings, please wait ",))
            wait_thread.start()
            run_generate_true_area()
            stop_event.set()
            wait_thread.join()

        elif choice == 3:
            wait_thread = threading.Thread(target=show_wait_message, args=("Running Mandelbrot convergence analysis, please wait ",))
            wait_thread.start()
            run_mset_statistic_and_plot()
            stop_event.set()
            wait_thread.join()

        elif choice == 4:
            wait_thread = threading.Thread(target=show_wait_message, args=("Running Mandelbrot convergence analysis for fixed sample size and varying iterations, please wait ",))
            wait_thread.start()
            run_mset_statistic_and_plot_fixed_sample_size()
            stop_event.set()
            wait_thread.join()

        elif choice == 5:
            wait_thread = threading.Thread(target=show_wait_message, args=("Running Mandelbrot generating statistic sample, please wait ",))
            wait_thread.start()
            run_statistic_sample_generate()
            stop_event.set()
            wait_thread.join()

        elif choice == 5:
            wait_thread = threading.Thread(target=show_wait_message, args=("Running Mandelbrot statistic metrics and plots, please wait ",))
            wait_thread.start()
            run_statistic_metric()
            stop_event.set()
            wait_thread.join()

        elif choice == 0:
            print("Exiting the program.")
            break
        else:
            print("Invalid choice, please select a valid option.")

if __name__ == "__main__":
    main_controller()

