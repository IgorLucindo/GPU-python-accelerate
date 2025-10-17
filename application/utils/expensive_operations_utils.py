from numba import cuda
import numpy as np
import cupy as cp
import math


def expensive_cpu_loop(x: np.ndarray, repeats: int = 20) -> np.ndarray:
    """
    Expensive operation per element: repeated trig + exp to simulate cost
    """
    out = np.empty_like(x)
    for i in range(x.size):
        v = x[i]
        s = 0
        # perform some repeated math to make it expensive
        for k in range(repeats):
            s += math.sin(v + k * 0.12345) ** 2 + math.cos(v - k * 0.54321) ** 2
            s += math.exp((v % 3.0) * 0.1) * 0.001
        out[i] = s
    return out


def expensive_cupy(x_cupy, repeats: int = 20):
    """
    Vectorized CuPy implementation using broadcasting of an index array
    """
    # We'll build an index axis of shape (repeats,) and compute contributions in a broadcasted way.
    # This keeps the implementation concise and fully GPU-executed.
    k = cp.arange(repeats, dtype=x_cupy.dtype).reshape((repeats,) + (1,) * x_cupy.ndim)
    v = x_cupy[cp.newaxis, ...]  # shape (repeats, n)
    res = cp.sin(v + k * 0.12345) ** 2 + cp.cos(v - k * 0.54321) ** 2
    res += cp.exp((v % 3.0) * 0.1) * 0.001
    # sum over repeats axis
    return cp.sum(res, axis=0)


@cuda.jit
def expensive_kernel(in_arr, out_arr, repeats):
    i = cuda.grid(1)
    if i < in_arr.size:
        v = in_arr[i]
        s = 0.0
        for k in range(repeats):
            s += (math.sin(v + k * 0.12345) ** 2 + math.cos(v - k * 0.54321) ** 2)
            s += math.exp((v % 3.0) * 0.1) * 0.001
        out_arr[i] = s

def expensive_numba(x_np, repeats=20):
    n = x_np.size
    d_in = cuda.to_device(x_np)
    d_out = cuda.device_array_like(d_in)
    threads = 256
    blocks = (n + threads - 1) // threads
    expensive_kernel[blocks, threads](d_in, d_out, repeats)
    cuda.synchronize()
    return d_out