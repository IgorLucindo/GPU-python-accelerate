from utils.cheap_operations_utils import *
from utils.expensive_operations_utils import *


def correctness_test(n=1000, repeats=10):
    """
    Small correctness test
    """
    print("Running correctness tests (small arrays) ...")

    rng = np.random.RandomState(42)
    x = rng.uniform(-1.0, 1.0, size=(n,)).astype(np.float64)
    dx = cp.array(x)

    # Cheap operations
    cpu_cheap = cheap_cpu_loop(x)
    cp_cheap = cheap_cupy(dx)
    nb_cheap = cheap_numba(x)
    assert np.allclose(cpu_cheap, cp.asnumpy(cp_cheap), rtol=1e-9, atol=1e-12)
    assert np.allclose(cpu_cheap, nb_cheap.copy_to_host(), rtol=1e-9, atol=1e-12)

    # Expensive operations
    cpu_exp = expensive_cpu_loop(x, repeats=repeats)
    cp_exp = expensive_cupy(dx, repeats=repeats)
    nb_exp = expensive_numba(x, repeats=repeats)
    assert np.allclose(cpu_exp, cp.asnumpy(cp_exp), rtol=1e-9, atol=1e-10)
    assert np.allclose(cpu_exp, nb_exp.copy_to_host(), rtol=1e-9, atol=1e-10)

    print("Correctness tests passed.\n")