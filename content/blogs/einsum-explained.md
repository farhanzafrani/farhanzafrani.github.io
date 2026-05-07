---
title: "Einstein Summation (einsum) for Multi-Dimensional Arrays"
date: "2024-10-20"
tags: [Python, NumPy, Linear Algebra]
excerpt: >
  einsum notation lets you write complex tensor contractions — matrix
  multiplications, batch operations, traces — in one readable line.
  Here is how to master it.
---

## Why einsum?

NumPy and PyTorch's `einsum` function lets you express any combination of
tensor contractions, transposes, and sums using a compact index notation
inspired by Einstein's summation convention.

Instead of:

```python
# Batched matrix multiply — verbose
result = torch.bmm(A, B)
```

You write:

```python
result = torch.einsum('bik,bkj->bij', A, B)
```

## Reading the Notation

The string `'bik,bkj->bij'` is read as:

| Part | Meaning |
|---|---|
| `bik` | First tensor has axes batch, i, k |
| `bkj` | Second tensor has axes batch, k, j |
| `->bij` | Output has axes batch, i, j |
| `k` absent in output | Sum over k (contraction) |

## Common Patterns

```python
# Matrix multiply
torch.einsum('ij,jk->ik', A, B)

# Batch matrix multiply
torch.einsum('bij,bjk->bik', A, B)

# Dot product
torch.einsum('i,i->', a, b)

# Outer product
torch.einsum('i,j->ij', a, b)

# Trace
torch.einsum('ii->', A)

# Transpose
torch.einsum('ij->ji', A)

# Element-wise then sum (like torch.sum(A * B))
torch.einsum('ij,ij->', A, B)
```

## Practical Example — Attention Scores

The scaled dot-product attention score matrix is a perfect einsum use case:

```python
# Q: (batch, heads, seq, d_k)
# K: (batch, heads, seq, d_k)
scores = torch.einsum('bhid,bhjd->bhij', Q, K) / d_k ** 0.5
```

Clean, explicit, and often faster than equivalent `torch.matmul` with
reshaping because einsum can fuse operations under the hood.

## Tips

- Shared indices that appear in both inputs but not the output are **contracted** (summed over).
- Indices only in one input are **free** — they pass through to the output.
- Use `opt_einsum` for optimal contraction order on large tensors.
