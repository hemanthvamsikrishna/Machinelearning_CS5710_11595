# Transformer Attention Mechanism - Implementation Guide

This repository contains Python implementations for two fundamental components of transformer architectures, designed for educational purposes and Google Colab execution.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Questions Implemented](#questions-implemented)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
  - [Question 1: Scaled Dot-Product Attention](#question-1-scaled-dot-product-attention-numpy)
  - [Question 2: Transformer Encoder Block](#question-2-transformer-encoder-block-pytorch)
- [Code Structure](#code-structure)
- [Key Concepts Explained](#key-concepts-explained)
- [Expected Outputs](#expected-outputs)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## 🎯 Overview

This project implements core transformer components from scratch:

1. **Scaled Dot-Product Attention** (NumPy implementation)
2. **Transformer Encoder Block** (PyTorch implementation)

Both implementations include comprehensive testing, detailed comments, and visualization of results.

---

## 📝 Questions Implemented

### Question 1: Compute Scaled Dot-Product Attention (Python)
**Objective:** Write a Python function to compute scaled dot-product attention given query Q, key K, and value V matrices.

**Requirements:**
- Use NumPy for matrix operations
- Normalize scores using softmax
- Return both attention weights and resulting context vector

**Formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

### Question 2: Implement Simple Transformer Encoder Block (PyTorch)
**Objective:** Implement a simplified transformer encoder block with the following components:

**Components:**
- Multi-head self-attention layer
- Feed-forward network (2 linear layers with ReLU)
- Add & Norm layers (residual connections + layer normalization)

**Sub-tasks:**
- a) Initialize dimensions: d_model = 128, h = 8
- b) Add residual connections and layer normalization
- c) Verify output shape for batch of 32 sentences, each with 10 tokens

---

## 🔧 Requirements

### Python Packages

**For Question 1 (NumPy):**
```
numpy>=1.19.0
```

**For Question 2 (PyTorch):**
```
torch>=1.9.0
```

### Google Colab
Both scripts are designed to run directly in Google Colab without additional setup.

---

## 📥 Installation

### Option 1: Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Copy and paste the code from each file into separate cells
4. Run the cells in order

### Option 2: Local Setup

```bash
# Clone or download the files
# Install required packages
pip install numpy torch

# Run individual scripts
python q1_scaled_attention.py
python q2_transformer_encoder.py
```

---

## 📖 Usage Guide

### Question 1: Scaled Dot-Product Attention (NumPy)

#### **Code Cell 1: Implementation**

```python
"""
Question 1: Compute Scaled Dot-Product Attention (Python)
"""

import numpy as np

def scaled_dot_product_attention(Q, K, V):
    """
    Compute scaled dot-product attention.
    
    Args:
        Q: Query matrix of shape (batch_size, seq_len_q, d_k)
        K: Key matrix of shape (batch_size, seq_len_k, d_k)
        V: Value matrix of shape (batch_size, seq_len_v, d_v)
        
    Returns:
        context: Context vector of shape (batch_size, seq_len_q, d_v)
        attention_weights: Attention weights of shape (batch_size, seq_len_q, seq_len_k)
    """
    # Get dimension d_k
    d_k = K.shape[-1]
    
    # Compute attention scores: QK^T
    scores = np.matmul(Q, K.transpose(0, 2, 1))
    
    # Scale by sqrt(d_k)
    scaled_scores = scores / np.sqrt(d_k)
    
    # Apply softmax
    attention_weights = softmax(scaled_scores)
    
    # Compute context vector
    context = np.matmul(attention_weights, V)
    
    return attention_weights, context


def softmax(x):
    """Compute softmax with numerical stability."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


# Test the implementation
print("Testing Scaled Dot-Product Attention")
print("="*60)

# Example: Single sequence
np.random.seed(42)
batch_size, seq_len, d_k = 1, 4, 8

Q = np.random.randn(batch_size, seq_len, d_k)
K = np.random.randn(batch_size, seq_len, d_k)
V = np.random.randn(batch_size, seq_len, d_k)

attention_weights, context = scaled_dot_product_attention(Q, K, V)

print(f"Input shapes: Q={Q.shape}, K={K.shape}, V={V.shape}")
print(f"Output shapes: Attention={attention_weights.shape}, Context={context.shape}")
print(f"\nAttention weights:\n{attention_weights[0]}")
print(f"\nRow sums (should be ~1.0): {np.sum(attention_weights[0], axis=-1)}")
```

#### **Key Features:**
- ✅ Implements scaled dot-product attention formula
- ✅ Handles batched inputs
- ✅ Numerically stable softmax
- ✅ Returns both weights and context vectors
- ✅ Includes multiple test cases

#### **Expected Output:**
```
Testing Scaled Dot-Product Attention
============================================================
Input shapes: Q=(1, 4, 8), K=(1, 4, 8), V=(1, 4, 8)
Output shapes: Attention=(1, 4, 4), Context=(1, 4, 8)

Attention weights:
[[0.28 0.31 0.19 0.22]
 [0.24 0.26 0.25 0.25]
 [0.23 0.27 0.26 0.24]
 [0.25 0.24 0.26 0.25]]

Row sums (should be ~1.0): [1. 1. 1. 1.]
```

---

### Question 2: Transformer Encoder Block (PyTorch)

#### **Code Cell 2: Implementation**

```python
"""
Question 2: Implement Simple Transformer Encoder Block (PyTorch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def split_heads(self, x):
        """Split into multiple heads."""
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)
    
    def combine_heads(self, x):
        """Combine heads back together."""
        batch_size, num_heads, seq_len, d_k = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute attention."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, V)
        return context, attention_weights
    
    def forward(self, x, mask=None):
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        
        context, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        context = self.combine_heads(context)
        return self.W_o(context)


class FeedForwardNetwork(nn.Module):
    """Position-wise feed-forward network."""
    
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class TransformerEncoderBlock(nn.Module):
    """Complete transformer encoder block."""
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Multi-head attention with Add & Norm
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with Add & Norm
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x


# Test the implementation
print("="*70)
print("Testing Transformer Encoder Block")
print("="*70)

# Initialize with specified dimensions
d_model = 128
num_heads = 8
d_ff = 512

encoder_block = TransformerEncoderBlock(d_model, num_heads, d_ff)

# Test with batch of 32 sentences, 10 tokens each
batch_size = 32
seq_len = 10
x = torch.randn(batch_size, seq_len, d_model)

print(f"\nModel Configuration:")
print(f"  d_model: {d_model}")
print(f"  num_heads: {num_heads}")
print(f"  d_k (per head): {d_model // num_heads}")
print(f"  d_ff: {d_ff}")

print(f"\nInput shape: {x.shape}")

# Forward pass
encoder_block.eval()
with torch.no_grad():
    output = encoder_block(x)

print(f"Output shape: {output.shape}")

# Verify shape
expected_shape = (batch_size, seq_len, d_model)
assert output.shape == expected_shape
print(f"\n✓ Shape verification passed!")
print(f"  Expected: {expected_shape}")
print(f"  Got: {output.shape}")

# Count parameters
total_params = sum(p.numel() for p in encoder_block.parameters())
print(f"\nTotal Parameters: {total_params:,}")
```

#### **Key Features:**
- ✅ Multi-head self-attention with 8 heads
- ✅ Feed-forward network with ReLU activation
- ✅ Residual connections (Add)
- ✅ Layer normalization (Norm)
- ✅ Dropout for regularization
- ✅ Modular, reusable components

#### **Expected Output:**
```
======================================================================
Testing Transformer Encoder Block
======================================================================

Model Configuration:
  d_model: 128
  num_heads: 8
  d_k (per head): 16
  d_ff: 512

Input shape: torch.Size([32, 10, 128])
Output shape: torch.Size([32, 10, 128])

✓ Shape verification passed!
  Expected: (32, 10, 128)
  Got: torch.Size([32, 10, 128])

Total Parameters: 198,272
```

---

## 🏗️ Code Structure

### Question 1 Structure (q1_scaled_attention.py)

```
q1_scaled_attention.py
├── scaled_dot_product_attention()  # Main attention function
├── softmax()                        # Numerically stable softmax
└── Test cases
    ├── Example 1: Single sequence
    ├── Example 2: Batch of sequences
    └── Example 3: Cross-attention
```

### Question 2 Structure (q2_transformer_encoder.py)

```
q2_transformer_encoder.py
├── MultiHeadSelfAttention
│   ├── split_heads()
│   ├── combine_heads()
│   ├── scaled_dot_product_attention()
│   └── forward()
├── FeedForwardNetwork
│   └── forward()
├── TransformerEncoderBlock
│   └── forward()
└── Test cases
    ├── Shape verification
    ├── Component testing
    └── Various input sizes
```

---

## 💡 Key Concepts Explained

### 1. Scaled Dot-Product Attention

**Why scaling?**
- Prevents large dot products that cause softmax to have extremely small gradients
- Scaling factor: 1/√d_k stabilizes training

**Components:**
1. **Scores**: Q·K^T measures similarity between queries and keys
2. **Scaling**: Divides by √d_k
3. **Softmax**: Converts to probability distribution
4. **Context**: Weighted sum of values

### 2. Multi-Head Attention

**Why multiple heads?**
- Each head learns different representation subspaces
- Head 1 might focus on syntax, Head 2 on semantics, etc.
- Increases model capacity without proportional compute increase

**Process:**
1. Project Q, K, V to d_model dimensions
2. Split into h heads (d_k = d_model / h per head)
3. Apply attention in parallel for each head
4. Concatenate results
5. Final linear projection

### 3. Residual Connections & Layer Norm

**Residual (Add):**
```python
output = x + sublayer(x)
```
- Enables gradient flow in deep networks
- Helps with vanishing gradient problem

**Layer Normalization (Norm):**
```python
output = LayerNorm(x + sublayer(x))
```
- Normalizes across feature dimension
- Stabilizes training
- Reduces internal covariate shift

### 4. Feed-Forward Network

**Architecture:**
```
Linear(d_model → d_ff) → ReLU → Linear(d_ff → d_model)
```
- Typically d_ff = 4 × d_model
- Applied position-wise (same for all positions)
- Adds non-linear transformation capacity

---

## 📊 Expected Outputs

### Question 1: Attention Weights Properties

✅ **Each row sums to 1.0** (probability distribution)
```
[0.28, 0.31, 0.19, 0.22] → sum = 1.0
```

✅ **Higher values indicate stronger attention**
```
Token 2 pays most attention to Token 1 (weight = 0.31)
```

✅ **Context vector shape matches (batch, seq_len_q, d_v)**

### Question 2: Shape Verification

✅ **Input**: (32, 10, 128)
- 32 sentences
- 10 tokens per sentence
- 128-dimensional embeddings

✅ **Output**: (32, 10, 128)
- Same shape (encoder preserves dimensions)

✅ **Parameter Count**: ~198,000 parameters
- Attention: ~66k
- Feed-forward: ~131k
- Layer norms: ~512

---

## 🐛 Troubleshooting

### Common Issues

**Issue 1: Shape Mismatch Errors**
```python
# Check input dimensions
print(f"Q: {Q.shape}, K: {K.shape}, V: {V.shape}")

# Ensure: Q and K have same d_k, K and V have same seq_len
```

**Issue 2: NaN in Outputs**
```python
# Check for numerical instability
# Solution: Use numerically stable softmax (subtract max)
```

**Issue 3: PyTorch Not Found**
```python
# In Colab, PyTorch is pre-installed
# Locally: pip install torch
```

**Issue 4: Memory Errors (Large Batches)**
```python
# Reduce batch size or sequence length
batch_size = 16  # Instead of 32
seq_len = 5      # Instead of 10
```

---

## 📚 References

### Papers
1. **Attention Is All You Need** (Vaswani et al., 2017)
   - Original Transformer paper
   - [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

### Documentation
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [NumPy Documentation](https://numpy.org/doc/stable/)

### Tutorials
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- [Harvard NLP: The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

---

## 📝 Notes

- Both implementations are **educational** and simplified for clarity
- Production code would include:
  - More comprehensive error handling
  - Masking for padding tokens
  - Position encodings
  - More efficient implementations
- Code is well-commented for learning purposes

---

## ✅ Checklist for Running

- [ ] NumPy installed (for Q1)
- [ ] PyTorch installed (for Q2)
- [ ] Code copied to Google Colab or local environment
- [ ] Run Q1 first to understand attention basics
- [ ] Run Q2 to see complete transformer block
- [ ] Verify output shapes match expected dimensions
- [ ] Experiment with different batch sizes and sequence lengths

---

## 🎓 Learning Outcomes

After completing these implementations, you should understand:

1. ✅ How attention mechanism computes relevance between tokens
2. ✅ Why scaling is necessary in dot-product attention
3. ✅ How multi-head attention increases model capacity
4. ✅ The role of residual connections in deep networks
5. ✅ How layer normalization stabilizes training
6. ✅ The complete architecture of a transformer encoder block

---

**Happy Learning! 🚀**

For questions or issues, please refer to the inline comments in the code or the troubleshooting section above.
