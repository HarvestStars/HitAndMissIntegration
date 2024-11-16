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
import seaborn as sns
import numpy as np

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
    if os.name == 'nt':
        num_workers = mp.cpu_count()
        Parallel(n_jobs=num_workers)(delayed(utils.mset_colors_parallel)(mandelbrotAnalysisPlatform, num_samples, max_iter) for num_samples, max_iter in mset_list)
    else:
        for num_samples, max_iter in mset_list:
            utils.mset_colors_parallel(mandelbrotAnalysisPlatform, num_samples, max_iter)

    # orthogonal sampling has to be run sequentially
    mandelbrotAnalysisPlatform._load_library()
    utils.mset_colors_ortho_seq(mandelbrotAnalysisPlatform, num_samples_list_perfect_root, max_iter_list)


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
    trueA = utils.read_area_from_file()
    if trueA == 0:
        trueA = utils.get_and_save_true_area(mandelbrotAnalysisPlatform)
    
    area_data_set = utils.read_area_series_from_files(mandelbrotAnalysisPlatform)

    # Check if data exists for all sampling methods, if not, generate and save it
    if not all(area_data_set[mandelbrotAnalysisPlatform.get_sample_name(sample_type)] for sample_type in [0, 1, 2]):
        print("Data not found, generating and saving data.")
        utils.save_area_series_into_files(mandelbrotAnalysisPlatform)
        area_data_set = utils.read_area_series_from_files(mandelbrotAnalysisPlatform)

    # Extract data for plotting
    num_samples_vals1, max_iter_vals1, area_vals1 = zip(*area_data_set["Pure"]) if area_data_set["Pure"] else ([], [], [])
    num_samples_vals2, max_iter_vals2, area_vals2 = zip(*area_data_set["LHS"]) if area_data_set["LHS"] else ([], [], [])
    num_samples_vals3, max_iter_vals3, area_vals3 = zip(*area_data_set["Ortho"]) if area_data_set["Ortho"] else ([], [], [])

    # Calculate differences from alpha
    area_diff_vals1 = [area - trueA for area in area_vals1]
    area_diff_vals2 = [area - trueA for area in area_vals2]
    area_diff_vals3 = [area - trueA for area in area_vals3]

    # store the image into a file, if no existing directory, create one
    os.makedirs(mandelbrot_analysis.IMG_CONVERGENCE_DIR, exist_ok=True)

    # Generate individual 3D plots
    utils.plot_individual_3d(num_samples_vals1, max_iter_vals1, area_vals1, 'b', 'o', 'Pure Random Sampling', f'{mandelbrot_analysis.IMG_CONVERGENCE_DIR}/3D_Diff_pure_random_sampling.png')
    utils.plot_individual_3d(num_samples_vals2, max_iter_vals2, area_vals2, 'r', '^', 'LHS Sampling', f'{mandelbrot_analysis.IMG_CONVERGENCE_DIR}/3D_Diff_lhs_sampling.png')
    utils.plot_individual_3d(num_samples_vals3, max_iter_vals3, area_vals3, 'g', 's', 'Orthogonal Sampling', f'{mandelbrot_analysis.IMG_CONVERGENCE_DIR}/3D_Diff_orthogonal_sampling.png')

    # Generate heatmaps
    utils.generate_heatmap(max_iter_vals1 , num_samples_vals1, area_vals1, "Heatmap - Pure Random Sampling Area", "Max Iterations", "Number of Samples", f'{mandelbrot_analysis.IMG_CONVERGENCE_DIR}/heatmap_pure_random_sampling.png')
    utils.generate_heatmap(max_iter_vals2 , num_samples_vals2, area_vals2, "Heatmap - LHS Sampling Area", "Max Iterations", "Number of Samples", f'{mandelbrot_analysis.IMG_CONVERGENCE_DIR}/heatmap_lhs_sampling.png')
    utils.generate_heatmap(max_iter_vals3 , num_samples_vals3, area_vals3, "Heatmap - Orthogonal Sampling Area", "Max Iterations", "Number of Samples", f'{mandelbrot_analysis.IMG_CONVERGENCE_DIR}/heatmap_orthogonal_sampling.png')

# -----------------------------------------------------------inverstigate convergence for fixed sample size------------------------------------
def run_mset_s_and_i_analysis():
    if mandelbrotAnalysisPlatform.lib is None:
        mandelbrotAnalysisPlatform._load_library()
    
    # Try to read data from files
    trueA = utils.read_area_from_file()
    if trueA == 0:
        trueA = utils.get_and_save_true_area(mandelbrotAnalysisPlatform)
    
    area_data_set = utils.read_area_series_from_files(mandelbrotAnalysisPlatform)

    # Extract data for plotting
    num_samples_vals1, max_iter_vals1, area_vals1 = zip(*area_data_set["Pure"]) if area_data_set["Pure"] else ([], [], [])
    num_samples_vals2, max_iter_vals2, area_vals2 = zip(*area_data_set["LHS"]) if area_data_set["LHS"] else ([], [], [])
    num_samples_vals3, max_iter_vals3, area_vals3 = zip(*area_data_set["Ortho"]) if area_data_set["Ortho"] else ([], [], [])

    # Calculate differences from alpha
    area_diff_vals1 = [area - trueA for area in area_vals1]
    area_diff_vals2 = [area - trueA for area in area_vals2]
    area_diff_vals3 = [area - trueA for area in area_vals3]

    # Generate convergence plots for each sampling method
    utils.plot_convergence_curve(num_samples_vals1, max_iter_vals1, area_diff_vals1, 'Pure Random Sampling', f'{mandelbrot_analysis.IMG_CONVERGENCE_DIR}/pure_random_sampling_convergence')
    utils.plot_convergence_curve(num_samples_vals2, max_iter_vals2, area_diff_vals2, 'LHS Sampling', f'{mandelbrot_analysis.IMG_CONVERGENCE_DIR}/lhs_sampling_convergence')
    utils.plot_convergence_curve(num_samples_vals3, max_iter_vals3, area_diff_vals3, 'Orthogonal Sampling', f'{mandelbrot_analysis.IMG_CONVERGENCE_DIR}/orthogonal_sampling_convergence')


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
    metrics.plot_confidence_intervals(confidence_intervals)
    metrics.plot_histograms()

    # Plot area distributions
    metrics.plot_area_distributions()

#------------------------------------------------------------improvement converge--------------------------------------------------------------
def run_improvement_converge():
    if mandelbrotAnalysisPlatform.lib is None:
        mandelbrotAnalysisPlatform._load_library()
    
    # Try to read data from files
    trueA = utils.read_area_from_file()
    if trueA == 0:
        trueA = utils.get_and_save_true_area(mandelbrotAnalysisPlatform)
    
    area_data_set = utils.read_area_series_from_files(mandelbrotAnalysisPlatform)

    # Check if data exists for all sampling methods, if not, generate and save it
    if not all(area_data_set[mandelbrotAnalysisPlatform.get_sample_name(sample_type)] for sample_type in [0, 1, 2]):
        print("Data not found, generating and saving data.")
        utils.save_area_series_into_files(mandelbrotAnalysisPlatform)
        area_data_set = utils.read_area_series_from_files(mandelbrotAnalysisPlatform)

    plane_area = abs((mandelbrotAnalysisPlatform.real_range[1] - mandelbrotAnalysisPlatform.real_range[0]) * (mandelbrotAnalysisPlatform.imag_range[1] - mandelbrotAnalysisPlatform.imag_range[0]))
    dimension_separate_number = 4
    partial_area = plane_area / dimension_separate_number**2
    adaptive_num_samples = []
    adaptive_iter_vals = []
    adaptive_areas = []

    try:
        with open(f'{mandelbrot_analysis.IMG_CONVERGENCE_IMPROVE_DIR}/mandelbrotArea_adaptive.txt', "r") as file:
            print("Reading data from mandelbrotArea_adaptive.txt")
            area_data_set["Adaptive"] = []
            for line in file:
                num_samples, max_iter, area = line.split()
                area_data_set["Adaptive"].append((int(num_samples), int(max_iter), float(area)))
                adaptive_num_samples.append(int(num_samples))
                adaptive_iter_vals.append(int(max_iter))
                adaptive_areas.append(float(area))

    except FileNotFoundError:
        # re create the data set
        num_samples_list_perfect_root = [500, 800, 1000, 1600, 2000, 2400, 2600, 3000]
        max_iter_list = [100, 150, 200, 240, 300, 400, 600, 700, 800, 900, 1000]
        mset_list = list(itertools.product(num_samples_list_perfect_root, max_iter_list))
        
        for num_samples_root, max_iter in mset_list:
            adaptive_area = 0
            adaptive_samples = mandelbrotAnalysisPlatform.adaptive_sampling(num_samples_root, dimension_separate_number)
            for adaptive_sample in adaptive_samples:
                if len(adaptive_sample) == 0:
                    continue
                area_partial = mandelbrotAnalysisPlatform.calcu_mandelbrot_area(adaptive_sample, max_iter, partial_area)
                adaptive_area += area_partial
            print(f"Area of the Mandelbrot set with method Adaptive, {num_samples_root**2} samples and {max_iter} max iterations, the area is {round(adaptive_area, 6)}")
            adaptive_num_samples.append(num_samples_root**2)
            adaptive_iter_vals.append(max_iter)
            adaptive_areas.append(round(adaptive_area, 6))
        # store the image into a file, if no existing directory, create one
        os.makedirs(mandelbrot_analysis.IMG_CONVERGENCE_IMPROVE_DIR, exist_ok=True)
        # Save pure random sampling data to file
        with open(f'{mandelbrot_analysis.IMG_CONVERGENCE_IMPROVE_DIR}/mandelbrotArea_adaptive.txt', "w") as file:
            for num_samples, max_iter, area in zip(adaptive_num_samples, adaptive_iter_vals, adaptive_areas):
                file.write(f"{num_samples} {max_iter} {area:.6f}\n")

    # Calculate differences from alpha
    area_diff_vals = [area - trueA for area in adaptive_areas]

    # Generate individual 3D plots
    utils.plot_individual_3d(adaptive_num_samples, adaptive_iter_vals, adaptive_areas, 'b', 'o', 'Adaptive Sampling', f'{mandelbrot_analysis.IMG_CONVERGENCE_IMPROVE_DIR}/3D_Diff_adaptive_sampling.png')

    # Generate heatmaps
    utils.generate_heatmap(adaptive_iter_vals, adaptive_num_samples, adaptive_areas, "Heatmap - Adaptive Sampling Area", "Max Iterations", "Number of Samples", f'{mandelbrot_analysis.IMG_CONVERGENCE_IMPROVE_DIR}/heatmap_adaptive_sampling.png')

    # Generate convergence plots for each sampling method
    utils.plot_convergence_curve(adaptive_num_samples, adaptive_iter_vals, area_diff_vals, 'Adaptive Sampling', f'{mandelbrot_analysis.IMG_CONVERGENCE_IMPROVE_DIR}/adaptive_sampling_convergence')

    # Compare the convergence of the adaptive sampling method with the other methods
    utils.plot_convergence_comparison(area_data_set, trueA, f'{mandelbrot_analysis.IMG_CONVERGENCE_IMPROVE_DIR}/improvement')

# -----------------------------------------------------------main controller process-----------------------------------------------------------
def main_controller():
    while True:
        print("*" * 80)
        print("Select an option to run:")
        print("1: Run Mandelbrot color plottings")
        print("2: Run Generate True Area")
        print("3: Run Mandelbrot area calculation for visualization")
        print("4: Run Mandelbrot convergence analysis for s and i")
        print("5: Run Mandelbrot statistic sample generate")
        print("6: Run Mandelbrot statistic metrics and plots")
        print("7: Run Mandelbrot statistic improvement converge")
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
            wait_thread = threading.Thread(target=show_wait_message, args=("Running Mandelbrot True value calculation, please wait ",))
            wait_thread.start()
            run_generate_true_area()
            stop_event.set()
            wait_thread.join()

        elif choice == 3:
            wait_thread = threading.Thread(target=show_wait_message, args=("Running Mandelbrot area calculation, please wait ",))
            wait_thread.start()
            run_mset_statistic_and_plot()
            stop_event.set()
            wait_thread.join()

        elif choice == 4:
            wait_thread = threading.Thread(target=show_wait_message, args=("Running Mandelbrot convergence analysis for s and i, please wait ",))
            wait_thread.start()
            run_mset_s_and_i_analysis()
            stop_event.set()
            wait_thread.join()

        elif choice == 5:
            wait_thread = threading.Thread(target=show_wait_message, args=("Running Mandelbrot generating statistic sample, please wait ",))
            wait_thread.start()
            run_statistic_sample_generate()
            stop_event.set()
            wait_thread.join()

        elif choice == 6:
            wait_thread = threading.Thread(target=show_wait_message, args=("Running Mandelbrot statistic metrics and plots, please wait ",))
            wait_thread.start()
            run_statistic_metric()
            stop_event.set()
            wait_thread.join()

        elif choice == 7:
            wait_thread = threading.Thread(target=show_wait_message, args=("Running Mandelbrot improvement converge, please wait ",))
            wait_thread.start()
            run_improvement_converge()
            stop_event.set()
            wait_thread.join()

        elif choice == 0:
            print("Exiting the program.")
            break
        else:
            print("Invalid choice, please select a valid option.")

if __name__ == "__main__":
    main_controller()

