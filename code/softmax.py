import triton
import triton.language as tl
import torch

@triton.jit
def softmax(X, stride_xm, stride_xn, Y, stride_ym, stride_yn, M, N):
  """
    X: input matrix
    stride_xm: stride to access the next row in X
    stride_xn: stride to access the next column in X
    Y: output matrix
    M: number of rows in X
    N: number of columns in X
  """
  # Get current program ID 
  m = tl.program_id(axis=0)

  # Define size of each block 
  BLOCK_SIZE = 1024

  # Create column offsets
  n = tl.arange(0, 1024)

  # Compute memory address of X
  X = X + m * stride_xm + n * stride_xn

  # Create mask to prevent out-of-bound access
  mask = n < N

  # Load X into SRAM
  x = tl.load(X, mask=mask, other=-float("inf"))

  # Substract mas from x for numerical stability
  z = x - tl.max(x, axis=0)
  num = tl.exp(z)

  # Compute denominator of softmax
  denom = tl.sum(num, axis=0)

  # Softmax operation
  y = num / denom
  Y = Y + m * stride_xm + n * stride_xn

  # Write the result back to HBM
  tl.store(Y, y, mask=mask)

# Allocate input tensor
X = torch.normal(0, 1, size=(583, 931), device='cuda')
Y = torch.empty_like(X)

grid = (X.shape[0], )
softmax[grid](X, X.stride(0), X.stride(1), 
        Y, Y.stride(0), Y.stride(1),
        X.shape[0], X.shape[1])