# Mandelbrot Set Sampling Project

## Overview
This project provides tools to generate sample points for the Mandelbrot set using different sampling techniques, including pure random sampling, Latin Hypercube Sampling (LHS), and orthogonal sampling. It uses a C library to perform efficient orthogonal sampling and provides Python bindings to utilize the generated points for further visualization and analysis.

## Project Structure
```
project_root/
├── images/                                    # visulization results
├── simulation_results                         # our simulation results
├── ortho-pack/
│   ├── lib/
│   │   ├── ortho_sampling_generate.dll        # Compiled library for Windows
│   │   ├── libortho_sampling_generate.so      # Compiled library for Linux
│   │   └── libortho_sampling_generate.dylib   # Compiled library for macOS, not implemented yet
│   ├── mt19937ar.c                            # MT19937 random number generator source
│   ├── ortho_sampling_generate.c              # Sampling generation source
│   ├── rand_support.c                         # Support functions for random number generation
│   └── *.h                                    # Header files for the C/C++ sources
├── src/
│   ├── main.py                                # Main Python script for executing the sampling
│   ├── mandelbrot_analysis.py                 # Class implementation for Mandelbrot analysis
│   ├── metrics.py                             # Some statistical function for analysis
│   └── utils.py                               # Some helpful function for analysis                              
├── README.md
├── Assignment 1 - MANDELBROT.pdf              # Assignment descripition
└── CMakeLists.txt                             # CMake build configuration file
```
## Usage
1. **Install Dependencies**:
   - Python 3.x
   - Required Python packages: `numpy`, `joblib`
   ```sh
   pip install numpy joblib
   ```

2. **Run the Main Script**:
   ```sh
   python src/main.py
   ```

   - This script will generate Mandelbrot set points using different sampling methods and output the analysis.

## Project Flow
### Python Integration
- The main Python script (`src/main.py`) uses the `MandelbrotAnalysis` class to generate points on the complex plane using different sampling methods.
- The generated shared library (`.dll`, or `.so`) is dynamically loaded using `ctypes` to call the underlying C functions for point generation.
- Python code supports multiple platforms and dynamically chooses which shared library to load based on the system type (Windows, or Linux).

Upon running `src/main.py`, the following options are presented:

1. **Run Mandelbrot color plottings**: Generate visualizations of the Mandelbrot set using different color mappings to highlight the structure of the set.

2. **Run Generate True Area**: Use a high sample size (9 million points) and high iteration count (1000 iterations) to estimate the true area of the Mandelbrot set with maximal accuracy.

3. **Run Mandelbrot area calculation for visualization**: Generate different types of plots (heatmaps, 3D plots, planar views) to visualize how the sampled area estimates vary with different sample sizes and iteration counts.

4. **Run Mandelbrot convergence analysis for s and i**: Analyze convergence by fixing either the sample size or iteration count and study the difference between calculated and true area values.

5. **Run Mandelbrot statistic sample generate**: Generate area distribution samples under specific parameters (optimal sample size and iteration count) to understand statistical characteristics.

6. **Run Mandelbrot statistic metrics and plots**: Compute statistical metrics such as confidence intervals (CI), mean squared error (MSE), and means, and conduct hypothesis testing to validate results.

7. **Run Mandelbrot statistic improvement converge**: Use adaptive resampling to accelerate convergence and compare with other sampling methods.

0. **Exit**: Exit the program.

### Workflow
1. **Initial Visualization**:
   - Generate Mandelbrot set visualizations for the region $[-2, 2]$ in both real and imaginary parts using pure random, LHS, and orthogonal sampling methods. Save the generated images.

2. **True Area Calculation**:
   - Use a high sample size of 9 million points and 1000 iterations to obtain an accurate estimation of the Mandelbrot set area. This serves as the reference value for subsequent analysis.
   
3. **Area Visualization with Different Parameters**:
   - Generate visualizations (heatmaps, 3D plots, planar views) of the Mandelbrot set area using different sample sizes and iteration counts.

4. **Convergence Analysis**:
   - Fix the sample size and vary iteration counts to study the relationship between the area difference (true vs calculated) and iteration count.
   - Fix the iteration count and vary the sample size to study the relationship between area difference and sample size.

5. **Statistical Sample Generation**:
   - Perform repeated simulations to generate area samples for fixed optimal parameters (sample size and iteration count). This helps in understanding the statistical distribution of area estimates.

6. **Statistical Analysis**:
   - Calculate confidence intervals (CI), mean squared error (MSE), and means for the generated area samples. Formulate and test hypotheses to validate convergence results.

7. **Accelerated Convergence with Adaptive Sampling**:
   - Implement adaptive resampling techniques to accelerate convergence, and compare the performance with other methods.

## About the Orthogonal Sampling Library
The orthogonal sampling library has already been generated and placed in the appropriate directory `ortho-pack/lib/`, so typically **you don't need to recompile it yourself**. However, if you wish to compile it or encounter issues due to platform-specific differences, the following guide will help you generate the dynamic/shared library (.dll, .so, or .dylib) based on your operating system.
To compile the library, CMake is used for cross-platform compatibility. This guide explains how to generate the dynamic/shared library (`.dll`, `.so`, or `.dylib`) based on your operating system.

### Compilation Steps
1. **Navigate to the project root directory**:
   ```sh
   cd path/to/project_root
   ```

2. **Run CMake to generate the build system**:
   ```sh
   cmake -B build -S .
   ```
   - The `-B build` argument specifies that the build output should be placed in a folder called `build`.
   - The `-S .` argument specifies that the source is the current directory.

3. **Build the library**:
   ```sh
   cmake --build build
   ```
   This will generate the appropriate shared library in the `ortho-pack/lib` folder depending on your system:
   - On Windows: `ortho_sampling_generate.dll`
   - On Linux: `libortho_sampling_generate.so`
   - On macOS: `libortho_sampling_generate.dylib`

4. **Set the Output Directory**
   In the `CMakeLists.txt` file, the output directory for the compiled libraries is set as follows:
   ```cmake
   set_target_properties(ortho_sampling_generate PROPERTIES
       LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/ortho-pack/lib
   )
   ```

## Common Issues
1. **Library Not Found**:
   - Ensure the compiled library (`.dll`, `.so`, `.dylib`) is located in the `ortho-pack/lib` directory.
   - Verify that the correct library for your system is present.

2. **Serialization Errors with `joblib`**:
   - When using `joblib` for parallel execution, make sure not to pass `ctypes` objects, as they cannot be serialized. Load the dynamic library inside the worker function.

## Future Improvements
- Extend support for more sampling methods.
- Optimize the C library for better performance in large-scale sampling.
- Add more visualizations and analysis for the generated Mandelbrot set points.

## Contributing
Feel free to submit pull requests or open issues if you encounter any problems or have suggestions for improvement.

## License
This project is open-sourced under the MIT License. See the [LICENSE](./LICENSE) file for details.

