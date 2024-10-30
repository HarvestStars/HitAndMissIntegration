# TODO_LIST

## Step 1: Implement Sampling Techniques

- **Objective**: Implement three different sampling techniques to generate 2D sample points for the Mandelbrot set analysis.
- **Tasks**:
  - Implement the following sampling functions, each of which takes real and imaginary component ranges as input and generates a set of 2D sample points:
    1. **Pure Random Sampling** (`pure_random_sampling`): Use uniform random distribution to generate 2D points within the specified real and imaginary ranges.
    2. **Latin Hypercube Sampling** (`latin_hypercube_sampling`): Use Latin hypercube sampling to ensure well-distributed samples in the parameter space.
    3. **Orthogonal Sampling** (`orthogonal_sampling`): Divide the sample space into smaller regions and generate samples from each to achieve orthogonality.
  - **Output**: Each function should return a set (or similar collection) of 2D sample points, implemented in Python.

## Step 2: Monte Carlo Integration Function

- **Objective**: Implement a Monte Carlo integration function to estimate the area of the Mandelbrot set using different sampling methods.
- **Task**:
  - Create a function named `MonteC_Estimate` that takes the following inputs:
    - **`n`**: Maximum number of iterations to determine if a point belongs to the Mandelbrot set.
    - **`s`**: A sample set obtained from Step 1.
  - **Output**: The function should return an estimate of the Mandelbrot set area based on the given sample set and iterations.

## Step 3: Convergence Analysis Function

- **Objective**: Analyze the convergence behavior of the Monte Carlo estimate by comparing the effect of increasing the number of iterations.
- **Tasks**:
  - Implement a function named `Metrics` that:
    - Takes the `MonteC_Estimate` function as input.
    - Uses a baseline sample set (`s`), a maximum iteration limit (`maxN`), and a minimum iteration limit (`minN`).
    - Generates a dataset of points representing the relationship between the number of iterations (`j`) and the difference between the estimated Mandelbrot area at iteration `j` and the estimate at `maxN`.
  - **Output**: The function should return this dataset, which can be used to construct a plot illustrating convergence behavior.

## Step 4: Comparison of Sampling Techniques

- **Objective**: Compare the convergence rates of different sampling methods.
- **Tasks**:
  - Apply `Metrics` to each sampling technique (pure random, Latin hypercube, orthogonal) and collect the results.
  - Plot the convergence behavior of the different sampling methods.
    - **Visualization**: Create 3D plots if needed to visualize the convergence behavior and how each sampling method affects the accuracy of the Monte Carlo estimate.

## Step 5: GPU Acceleration for Monte Carlo Estimation

- **Objective**: Speed up the computation of the Monte Carlo estimate using GPU.
- **Tasks**:
  - Implement a GPU-accelerated version of the `MonteC_Estimate` function.
  - Utilize libraries such as **CUDA** (e.g., using `Numba` or `CuPy` in Python) to parallelize the computation.
  - Measure and compare the runtime performance of the GPU implementation versus the CPU implementation to demonstrate the speedup.

---

**Notes**:
- Each step should be verified and tested individually before proceeding to the next.
- Focus on modularizing the code to allow reuse and maintainability.
- For visualization, consider using libraries such as **Matplotlib** for 2D and 3D plotting.
- GPU acceleration may require adjustments to the sampling and iteration process to fully leverage parallelism.