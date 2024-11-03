# Mandelbrot Set Sampling Project

## Overview
This project provides tools to generate sample points for the Mandelbrot set using different sampling techniques, including pure random sampling, Latin Hypercube Sampling (LHS), and orthogonal sampling. It uses a C library to perform efficient orthogonal sampling and provides Python bindings to utilize the generated points for further visualization and analysis.

## Project Structure
```
project_root/
├── images/                                    # visulization results
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
│   └── mandelbrot_analysis.py                 # Class implementation for Mandelbrot analysis
├── todo.md                                    
├── README.md
├── Assignment 1 - MANDELBROT.pdf              # Assignment descripition
└── CMakeLists.txt                             # CMake build configuration file
```

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

## Project Flow
### Python Integration
- The main Python script (`src/main.py`) uses the `MandelbrotAnalysis` class to generate points on the complex plane using different sampling methods.
- The generated shared library (`.dll`, `.so`, or `.dylib`) is dynamically loaded using `ctypes` to call the underlying C functions for point generation.
- Python code supports multiple platforms and dynamically chooses which shared library to load based on the system type (Windows, Linux, or macOS).

### Workflow
1. **Sampling Generation**:
   - Sampling methods (`pure_random_sampling`, `latin_hypercube_sampling`, `orthogonal_sampling`) generate complex plane points.
   - Sampling is executed through functions defined in the C shared library for efficiency.

2. **Parallel and Sequential Processing**:
   - **Parallel Processing**: `joblib` is used for parallel Mandelbrot analysis using methods that can be parallelized without serializing pointers (like pure Python-based sampling).
   - **Sequential Processing**: Methods involving `ctypes` are processed sequentially to avoid serialization issues with pointers.

3. **Results**:
   - The generated points are used to evaluate whether they belong to the Mandelbrot set, and the results are visualized or further analyzed.

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
This project is open-sourced under the MIT License.

