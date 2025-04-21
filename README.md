# Triton Flash Attention Implementation

This project implements a memory-efficient Flash Attention mechanism using Triton programming language developed by OpenAI. Flash Attention is an optimization technique that reduces memory usage and improves computational efficiency in transformer models by avoiding materializing the full attention matrix.

## Project Description

The implementation includes:
- A Triton kernel for efficient attention computation
- A PyTorch module wrapper for easy integration
- Benchmarking tools to compare performance with standard attention
- Test suite to verify correctness against standard attention implementation

The code demonstrates how to implement attention mechanisms that are both memory-efficient and numerically stable, while maintaining accuracy comparable to standard attention implementations.

## Setup Instructions

1. Ensure you have a CUDA-capable GPU (NVIDIA A100 or similar recommended)
2. Install the required dependencies:
   ```bash
   pip install torch triton
   ```
3. Make sure you have NVIDIA drivers and CUDA toolkit installed (tested with CUDA 12.4)

## Running the Project

1. Open the Jupyter notebook `code/triton.ipynb`
2. Run all cells in sequence
3. The notebook will:
   - Verify CUDA availability
   - Define the Flash Attention implementation
   - Run tests comparing standard and flash attention
   - Perform benchmarking

## Expected Output

When you run the test function, you should see output similar to:

```
Testing with a batch size of 4, number of attention heads 8, sequence length 512 and head dimension 32.
Computing Standard Attention
Computing Flash Attention
Max difference between flash attention and standard attention: 1.4901161193847656e-06
Results match within error tolerance.

----- Performance Benchmark -----
Standard attention time: 0.0004 s
Flash attention time: 4.3755 s
Speedup: 0.00x
```

### Results Summary

- The implementation verifies numerical correctness by comparing outputs with standard attention
- The maximum difference between flash attention and standard attention should be very small (< 1e-4)
- Performance metrics show the relative speed of both implementations
- Note: Initial runs may be slower due to JIT compilation of the Triton kernel

## Notes

- The implementation includes dropout support in the interface, though the current version doesn't implement it
- The code uses block-wise computation to avoid materializing the full attention matrix
- Numerical stability is maintained through careful handling of the softmax computation 