from classes.timer import *
from utils.cheap_operations_utils import *
from utils.expensive_operations_utils import *
from utils.real_problem_operations_utils import *


def main():
    """
    Running benchmarks
    """
    timer = Timer()
    rng = np.random.RandomState(0)

    # -------- Cheap op benchmark --------
    n = int(5e6)
    x = rng.randn(n).astype(np.float64)
    dx = cp.array(x)

    timer.mark()
    # CPU loop
    _ = cheap_cpu_loop(x)
    timer.mark()
    # Cupy loop
    _ = cheap_cupy(dx)
    timer.mark()
    # Numba loop
    _ = cheap_numba(x).copy_to_host()
    cuda.synchronize()
    timer.mark()
    timer.update()

    print(
        f"CHEAP OP: n = {n}\n"
        f"  CPU loop: {timer.runtimes[0]:.4f} s\n"
        f"  CuPy vectorized: {timer.runtimes[1]:.4f} s\n"
        f"  Numba CUDA kernel: {timer.runtimes[2]:.4f} s (kernel + copy back)\n"
    )


    # -------- Expensive op benchmark --------
    n = int(1e6)
    repeats = 40
    x = rng.uniform(-2.0, 2.0, size=(n,)).astype(np.float64)
    dx = cp.array(x)

    timer.reset()
    timer.mark()
    # CPU loop
    _ = expensive_cpu_loop(x, repeats=repeats)
    timer.mark()
    # Cupy loop
    _ = expensive_cupy(dx, repeats=repeats)
    timer.mark()
    # Numba loop
    _ = expensive_numba(x, repeats=repeats).copy_to_host()
    cuda.synchronize()
    timer.mark()
    timer.update()

    print(
        f"EXPENSIVE OP: n = {n}, repeats = {repeats}\n"
        f"  CPU loop: {timer.runtimes[0]:.4f} s\n"
        f"  CuPy vectorized: {timer.runtimes[1]:.4f} s\n"
        f"  Numba CUDA kernel: {timer.runtimes[2]:.4f} s (kernel + copy back)\n"
    )


    # -------- Real problem: pairwise distances --------
    M = 800
    N = 800
    D = 64
    A = rng.randn(M, D).astype(np.float64)
    B = rng.randn(N, D).astype(np.float64)

    timer.reset()
    timer.mark()
    # CPU loop
    _ = pairwise_distances_cpu(A, B)
    timer.mark()
    # Cupy loop
    _ = pairwise_distances_cupy(cp.array(A), cp.array(B))
    timer.mark()
    # Numba loop
    _ = pairwise_numba(A, B).copy_to_host()
    cuda.synchronize()
    timer.mark()
    timer.update()

    print(
        f"PAIRWISE DISTANCES: M={M}, N={N}, D={D} -> out shape ({M},{N})\n"
        f"  CPU loop: {timer.runtimes[0]:.4f} s\n"
        f"  CuPy vectorized: {timer.runtimes[1]:.4f} s\n"
        f"  Numba CUDA kernel: {timer.runtimes[2]:.4f} s (kernel + copy back)\n"
    )


if __name__ == "__main__":
    main()