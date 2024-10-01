---
layout: post
title: Understanding Self-Attention in Transformers
date: 2024-09-01
description: understanding self-attension mechanism in transformers in details
tags: self_attention
thumbnail: assets/img/transformer_with_pytorch.png
categories: transformers , deeplearnings, sequence models
featured: true
toc:
  beginning: true
---

============================================

<img src="/assets/img/transformer_with_pytorch.png" alt="transformer" width="800"/>


Self-Attention is a fundamental mechanism in modern Natural Language Processing (NLP) models, especially in Transformer architectures. It allows models to weigh the importance of different words in a sequence when encoding a particular word, enabling the capture of contextual relationships effectively.

This guide provides a detailed understanding of self-attention, its necessity, and how it works, along with illustrative examples and code snippets.

## Introduction to Word Embeddings
-----------------------------------

**Word Embeddings** are numerical representations of words in a continuous vector space where semantically similar words are mapped closely together. They have been instrumental in capturing the meaning and context of words in various NLP tasks.

### Example: Word2Vec Embeddings

Word2Vec is a popular technique for generating word embeddings by training neural networks on large corpora.

```python
# Example of obtaining word embeddings using gensim
from gensim.models import Word2Vec

sentences = [["king", "queen", "man", "woman"],
             ["apple", "orange", "fruit", "banana"]]

model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=2)
vector = model.wv['king']  # Retrieves the embedding for 'king'
```

**Properties of Word Embeddings:**

-   Capture semantic relationships (e.g., `king` - `man` + `woman` ≈ `queen`).
-   Represent words in fixed-size dense vectors.

* * * * *

## Limitations of Traditional Word Embeddings
----------------------------------------------

Despite their effectiveness, traditional word embeddings have several limitations:

### 2.1. Contextual Insensitivity

-   **Static Representations:** Each word has a single embedding regardless of context.
-   **Ambiguity Handling:** Words with multiple meanings (polysemy) are not adequately represented.

**Example:**

Consider the word *"apple"* in different contexts:

1.  *"I ate an apple for breakfast."* (referring to the fruit)
2.  *"Apple released a new iPhone model."* (referring to the tech company)

Traditional embeddings assign the same vector to *"apple"* in both contexts, failing to capture the different meanings.

### 2.2. Inefficient Capture of Long-Range Dependencies

-   Difficulty in modeling relationships between distant words in a sequence.
-   Challenges in understanding complex sentence structures.

**Need for Contextual Representations:** To overcome these limitations, models require mechanisms that can dynamically adjust word representations based on context. This is where **Self-Attention** comes into play.

* * * * *

3\. What is Self-Attention?
---------------------------

**Self-Attention** is a mechanism that allows a model to focus on different parts of the input sequence when encoding a particular element. It computes a representation of each word by considering its relationship with other words in the sequence.

**Key Characteristics:**

-   **Contextual Understanding:** Adjusts word representations based on surrounding words.
-   **Parallel Computation:** Enables efficient processing of sequences.
-   **Flexibility:** Captures both short and long-range dependencies effectively.

**High-Level Idea:** For each word in a sequence, self-attention computes attention weights indicating the importance of other words in understanding the current word.

**Visual Representation:**


```bash
Input Sequence: [The, cat, sat, on, the, mat]

Self-Attention computes:
- For 'cat', higher attention to 'sat' and 'mat'
- For 'mat', higher attention to 'on' and 'cat'
```

* * * * *

## How Does Self-Attention Work?
---------------------------------

### 4.1. The Concept

Self-Attention involves three main components for each word/token in the sequence:

1.  **Query (Q):** Represents the current word we're focusing on.
2.  **Key (K):** Represents all other words in the sequence.
3.  **Value (V):** Contains information to be aggregated based on attention weights.

**Process Overview:**

-   Compute dot products between the Query and all Keys to get attention scores.
-   Apply softmax to obtain attention weights.
-   Multiply attention weights with the corresponding Values.
-   Sum the results to get the new representation for the current word.

### 4.2. Mathematical Formulation

Given an input sequence of embeddings X∈Rn×dX \in \mathbb{R}^{n \times d}X∈Rn×d:

1.  **Linear Transformations:**

    -   Q=XWQQ = XW^QQ=XWQ
    -   K=XWKK = XW^KK=XWK
    -   V=XWVV = XW^VV=XWV
    -   Where:
        -   WQ,WK,WV∈Rd×dkW^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}WQ,WK,WV∈Rd×dk​ are learnable weight matrices.
        -   nnn is the sequence length.
        -   ddd is the embedding dimension.
        -   dkd_kdk​ is the dimension of queries and keys.
2.  **Attention Scores:**

    -   Attention(Q,K,V)=softmax(QKTdk)V\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) VAttention(Q,K,V)=softmax(dk​​QKT​)V

**Explanation:**

-   **Dot Product (QK^T):** Measures similarity between queries and keys.
-   **Scaling (dk\sqrt{d_k}dk​​):** Prevents large dot product values leading to small gradients.
-   **Softmax:** Converts scores to probabilities representing attention weights.
-   **Weighted Sum (V):** Aggregates values based on attention weights to form contextualized embeddings.

### 4.3. Scaled Dot-Product Attention

**Formula:**

Attention(Q,K,V)=softmax(QKTdk)V\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) VAttention(Q,K,V)=softmax(dk​​QKT​)V

**Components Explained:**

-   **Q (Query):** Captures what we're searching for.
-   **K (Key):** Captures what information we have.
-   **V (Value):** Contains the actual information.

**Why Scale by dk\sqrt{d_k}dk​​?**

-   To counteract the effect of high-dimensional dot products, which can result in extremely large values, leading to vanishing gradients during training.

**Illustrative Diagram:**

```css
[Input Embeddings] --> [Q, K, V Linear Layers] --> [Compute Attention Scores] --> [Weighted Sum with V] --> [Output Embeddings]
```

* * * * *

## Implementing Self-Attention in Code
---------------------------------------

We'll implement both single-head and multi-head self-attention mechanisms using PyTorch.

### 5.1. Single-Head Self-Attention

**PyTorch Implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size

        self.query = nn.Linear(embed_size, embed_size)
        self.key   = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        # x: [batch_size, seq_length, embed_size]
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.bmm(Q, K.transpose(1, 2))  # [batch_size, seq_length, seq_length]
        attention_scores = attention_scores / (self.embed_size ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)

        out = torch.bmm(attention_weights, V)  # [batch_size, seq_length, embed_size]

        return out
```

**Usage Example:**


```python
batch_size = 2
seq_length = 5
embed_size = 64

x = torch.randn(batch_size, seq_length, embed_size)
self_attention = SelfAttention(embed_size)
output = self_attention(x)

print(output.shape)  # Output: torch.Size([2, 5, 64])`
```

**Explanation:**

-   **Input:** Batch of sequences with embeddings.
-   **Output:** Contextualized embeddings where each position considers other positions in the sequence.

### 5.2. Multi-Head Self-Attention

**Why Multi-Head?**

-   Allows the model to jointly attend to information from different representation subspaces at different positions.

**PyTorch Implementation:**

```python
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_size % num_heads == 0, "Embedding size must be divisible by num_heads"

        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads

        self.query = nn.Linear(embed_size, embed_size)
        self.key   = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        N, seq_length, _ = x.shape

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Reshape for multi-head
        Q = Q.view(N, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        K = K.view(N, seq_length, self.num_heads, self.head_dim).transpose(1,2)
        V = V.view(N, seq_length, self.num_heads, self.head_dim).transpose(1,2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        out = torch.matmul(attention_weights, V)
        out = out.transpose(1,2).contiguous().view(N, seq_length, self.embed_size)

        out = self.fc_out(out)

        return out
```

**Usage Example:**

```python
batch_size = 2
seq_length = 5
embed_size = 64
num_heads = 8

x = torch.randn(batch_size, seq_length, embed_size)
multi_head_attention = MultiHeadSelfAttention(embed_size, num_heads)
output = multi_head_attention(x)

print(output.shape)  # Output: torch.Size([2, 5, 64])
```

**Explanation:**

-   **Splitting into Heads:** The embedding is divided into multiple heads, allowing the model to focus on different parts of the input.
-   **Parallel Attention:** Each head performs self-attention in parallel.
-   **Concatenation and Projection:** The outputs from all heads are concatenated and passed through a final linear layer.

* * * * *

## Benefits of Self-Attention
------------------------------

-   **Contextual Representations:** Captures the context of words dynamically, handling polysemy effectively.
-   **Parallelism:** Unlike recurrent models, self-attention allows parallel computation over sequence positions, leading to faster training.
-   **Long-Range Dependencies:** Efficiently models relationships between distant words in a sequence.
-   **Scalability:** Performs well on long sequences without significant computational overhead.
-   **Versatility:** Applicable beyond NLP, including computer vision and speech processing tasks.

**Comparison with Traditional Models:**

| Aspect | RNNs/LSTMs | Self-Attention |
| --- | --- | --- |
| Computation Parallelism | Sequential | Parallel |
| Long-Range Dependencies | Challenging to capture | Efficiently captured |
| Contextual Understanding | Limited by sequence length | Comprehensive across entire sequence |
| Computational Efficiency | Less efficient for long sequences | More efficient and scalable |

* * * * *

## Conclusion
--------------

Self-Attention has revolutionized the field of NLP by providing a powerful mechanism to capture contextual information effectively. It addresses the limitations of traditional word embeddings and recurrent models, enabling the development of sophisticated models like Transformers that have set new benchmarks across various tasks.

Understanding self-attention is crucial for leveraging modern NLP architectures and for further innovation in the field.

* * * * *

## References
--------------

1.  Vaswani, A., et al. (2017). [**"Attention is All You Need"**](https://arxiv.org/abs/1706.03762). *Advances in Neural Information Processing Systems*.
2.  Devlin, J., et al. (2018). [**"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"**](https://arxiv.org/abs/1810.04805).
3.  **"The Illustrated Transformer"** by Jay Alammar.
4.  **PyTorch Documentation**.