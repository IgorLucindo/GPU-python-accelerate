# GPU-Python-Accelerate

## 🚀 Overview
This project studies three approaches for applying elementwise or pairwise computations:

1. **CPU** — single-threaded Python loop (simple, easy to reason about).  
2. **CuPy** — vectorized GPU implementation using CuPy arrays and broadcasting (leverages GPU behind-the-scenes).  
3. **Numba CUDA** — explicitly written CUDA kernel using Numba to parallelize loops and perform the per-element computation on the GPU.

We apply those 3 approaches to:
- A **cheap** elementwise operation (very little work per element).  
- An **expensive** elementwise operation (many floating operations per element to simulate heavier compute).  
- A **real-world** example: pairwise Euclidean distances between two sets (O(M × N × D) cost).  

The repository contains `main.py` which implements and times each method.

---

## 🧠 Notes

### What are the three approaches?
- **CPU**: a plain Python loop over elements. Good as a baseline and for small problem sizes or when GPU/tooling isn't available.  
- **CuPy**: write vectorized array expressions (very similar API to NumPy). CuPy executes operations on GPU and uses efficient kernels where possible. Very concise and usually the easiest route to GPU acceleration.  
- **Numba CUDA**: write explicit CUDA kernels via `numba.cuda.jit`. Gives precise control of thread/block layout and can achieve high performance when kernels are well-optimized.  

### When is it a good idea to parallelize?
- When the work per element is sizable (computational intensity) or when the data is large enough that copying to/from GPU is amortized by computation.  
- When operations are embarrassingly parallel (no or little cross-element dependency).  
- When latency is not critical but throughput is (GPU excels at throughput).  

### Can the problem be so big that we cannot parallelize everything at once?
- Yes. GPUs have finite memory. If your dataset doesn't fit device memory, options include:
  - Process in *chunks/batches* that fit GPU memory.
  - Use streaming (overlap host↔device transfers with computation).
  - Use distributed GPU solutions (multi-GPU).  
- Also, very large problems may require algorithmic changes (tiling, blocking) to achieve good performance.

### Best scenario for each approach
| Approach | Best Scenario |
|-----------|----------------|
| **CPU** | Small data sizes, debugging, or when GPU is unavailable |
| **CuPy** | Easy vectorizable operations, concise code, and built-in GPU routines |
| **Numba CUDA** | Fine-grained control, custom kernels, complex parallelization, or kernel optimization |

### Other notes and pitfalls
- **Data transfer overhead**: moving arrays between host and device is not free. Keep data on GPU between operations when possible.  
- **Precision & determinism**: expect small floating-point differences between CPU and GPU results — use `allclose` with tolerance.  
- **Synchronization**: profiling/timing must explicitly synchronize GPU before measuring elapsed time.  
- **Memory alignment & types**: ensure consistent dtypes (`float32` preferred for speed and memory).  
- **Kernel tuning**: adjust block/thread sizes, use shared memory, and avoid divergent branches for performance.

---

## 📦 Requirements
- Python 3.8+ (3.9/3.10/3.11 recommended)
- A CUDA-capable GPU and appropriate drivers
- CUDA toolkit
- Required:
  - `numpy`
  - `cupy` (must match your CUDA toolkit version)
  - `numba` (with CUDA support)

---

## 🛠️ Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

## ▶️ Execution

Run the benchmark:
```bash
python application/main.py
```