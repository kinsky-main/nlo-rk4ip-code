<!-- Plans for FFT -->

# General Problem

This is the primary module for performing Fast Fourier Transforms (FFT) and Inverse Fast Fourier Transforms (IFFT) on the simulation data. The issues to address include:
- Having multiple GPU backends (CUDA, OpenCL) to allow for flexibility in hardware usage.
- Ensuring that all operations stay on device memory to maximize performance.
- Is dependant on numerics and state modules having all the required functions and structures available in the respective backends.

# Proposed Solution

- Have a single fft.h header that exposes a public API for FFT operations.
- Implement backend-specific source files (fft_cuda.c, fft_opencl.c) that contain the actual implementations for each backend.
- Use conditional compilation to include the appropriate backend implementation based on user configuration.
- The public API will include functions to initialize the FFT plan for all steps of the simulation (ideally the size of the time grid does not change during the simulation), execute FFT and IFFT operations, and clean up resources.
- Ensure that all FFT operations are performed on device memory, with no unnecessary transfers to host memory.