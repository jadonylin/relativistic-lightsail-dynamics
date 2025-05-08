import numpy as np
import time
import torch

N = np.random.randn(4,4)
T = torch.rand(4,4)

print(N)
print(T)

n = 100000

t0 = time.time()
for i in range(n):
    eigvals, eigvecs = np.linalg.eig(N)
t1 = time.time()
print(f"Numpy eig time (average over {n} calculations): {(t1-t0)/n}")

t0 = time.time()
for i in range(n):
    eigvals = np.linalg.eigvals(N)
t1 = time.time()
print(f"Numpy eigvals time (average over {n} calculations): {(t1-t0)/n}")

t0 = time.time()
for i in range(n):
    eigvals, eigvecs = torch.linalg.eig(T)
t1 = time.time()
print(f"Torch eig time (average over {n} calculations): {(t1-t0)/n}")

t0 = time.time()
for i in range(n):
    eigvals = torch.linalg.eigvals(T)
t1 = time.time()
print(f"Torch eigvals time (average over {n} calculations): {(t1-t0)/n}")