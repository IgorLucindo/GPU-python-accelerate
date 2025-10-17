
from numba import cuda
import numpy as np


def cheap_cpu_loop(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    for i in range(x.size):
        # a light per-element op
        out[i] = (x[i] + 3.14159) * 0.5
    return out


def cheap_cupy(x_cupy):
    # vectorized CuPy expression
    return (x_cupy + 3.14159) * 0.5


# Numba kernel for cheap operation (elementwise)
@cuda.jit
def cheap_kernel(in_arr, out_arr):
    i = cuda.grid(1)
    if i < in_arr.size:
        out_arr[i] = (in_arr[i] + 3.14159) * 0.5

def cheap_numba(x_np):
    """
    Move numpy array to device, run kernel, and return device array
    """
    n = x_np.size
    d_in = cuda.to_device(x_np)
    d_out = cuda.device_array_like(d_in)
    threads = 256
    blocks = (n + threads - 1) // threads
    cheap_kernel[blocks, threads](d_in, d_out)
    cuda.synchronize()
    return d_out  # device array (numba/cuda device array)