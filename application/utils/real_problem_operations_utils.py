from numba import cuda
import numpy as np
import cupy as cp
import math


# Real problem: pairwise Euclidean distances (M x D) x (N x D)
def pairwise_distances_cpu(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # A: (M, D), B: (N, D) -> returns (M, N)
    M, D = A.shape
    N = B.shape[0]
    out = np.empty((M, N), dtype=A.dtype)
    for i in range(M):
        ai = A[i]
        for j in range(N):
            bj = B[j]
            s = 0.0
            for d in range(D):
                diff = ai[d] - bj[d]
                s += diff * diff
            out[i, j] = math.sqrt(s)
    return out


def pairwise_distances_cupy(A_cp, B_cp):
    # Use broadcasting: (M,1,D) - (1,N,D) -> (M,N,D), then sum over D, sqrt
    diff = A_cp[:, cp.newaxis, :] - B_cp[cp.newaxis, :, :]
    sq = diff ** 2
    s = cp.sum(sq, axis=2)
    return cp.sqrt(s)


@cuda.jit
def pairwise_kernel(A, B, out):
    # out shape (M, N)
    i, j = cuda.grid(2)
    M, D = A.shape
    N = B.shape[0]
    if i < M and j < N:
        s = 0.0
        for d in range(D):
            diff = A[i, d] - B[j, d]
            s += diff * diff
        out[i, j] = math.sqrt(s)

def pairwise_numba(A_np, B_np):
    M, D = A_np.shape
    N = B_np.shape[0]
    d_A = cuda.to_device(A_np)
    d_B = cuda.to_device(B_np)
    d_out = cuda.device_array((M, N), dtype=A_np.dtype)
    threads_per_block = (16, 16)
    blocks_x = (M + threads_per_block[0] - 1) // threads_per_block[0]
    blocks_y = (N + threads_per_block[1] - 1) // threads_per_block[1]
    pairwise_kernel[(blocks_x, blocks_y), threads_per_block](d_A, d_B, d_out)
    cuda.synchronize()
    return d_out