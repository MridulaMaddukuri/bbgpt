"""
Understanding self attention

"""

import torch
import torch.nn as nn  # neural network module
from torch.nn import functional as F

torch.manual_seed(1447)

# A math trick in self-attention
B, T, C = 4, 8, 2  # batch, time, channels
# enable token communicating with all the past tokens for a better model
x = torch.randn(B, T, C)
print(x.shape)
print(x[0])
# token in 5th location should get communicate with tokens in 1 to 4th locations, not anything beyong 5.
# simplest way to establish this is: For every t'th token, take an average of the channels from preceeding tokens from 1 to t (including t'th token)


# Version 1
xbow = torch.zeros((B, T, C))
for b in range(B):
    for t in range(T):
        xprev = x[b, : t + 1]  # (t,C)
        xbow[b, t] = torch.mean(xprev, 0)
print(xbow.shape)

print(xbow[0])

# x and xbow have the same size

# Version 2: Using matrix multiplication for weighted avg aggregation
# Make this more efficient by using matrix multiplication
print(torch.tril(torch.ones(T, T)))  # lower trianglular matrix
print(torch.tril(torch.ones(T, T)) @ x[0])  # cumulative sum
# but we need avg, so normalize the lower triangular matrix

lower_triangular_matrix = torch.tril(torch.ones(T, T))
# normalize so rows add up to 1
print(torch.sum(lower_triangular_matrix, axis=1))
print(torch.sum(lower_triangular_matrix, axis=1, keepdim=True))
lower_triangular_matrix_normalized = lower_triangular_matrix / torch.sum(
    lower_triangular_matrix, axis=1, keepdim=True
)
print(lower_triangular_matrix_normalized)

# batch matrix multiply. Each batch will have separate matric multiplication like in a for loop
xbow2 = lower_triangular_matrix_normalized @ x

print(xbow[0], xbow2[0])


# Version 3:  using softmax to bypass normalization
lower_triangular_matrix = torch.tril(torch.ones(T, T))
# will use this as a weight matrix that will establish the connection strength between tokens
wei = torch.zeros((T, T))  # weights here begin with 0
wei = wei.masked_fill(lower_triangular_matrix == 0, float("-inf"))
print(
    "Now notice that the softmax of wei is the same as lower triangular matrix normalized"
)
wei = F.softmax(wei, dim=1)
print(wei)

xbow3 = wei @ x

print(torch.allclose(xbow3[0], xbow[0]))


# Version 4: Self attention
# Every single token in the sequence will emit two vectors - a Query and a Key
# The Query vector - What am i looking for
# Key vector - What do I contain
# the dot product of Query and Key becomes Wei (contains the affinities among tokens)

# Single head of self attention ( a communication mechanism)
head_size = 16
key = nn.Linear(C, head_size, bias=False)  # what the token has
query = nn.Linear(C, head_size, bias=False)  # what the token is looking for
value = nn.Linear(C, head_size, bias=False)  # what the token is communicating to you
k = key(x)  # (B, T, head_size)
q = query(x)  # (B, T, head_size)
wei = q @ k.transpose(-2, -1)  # (B, T, 16) @ (B, 16, T) ---> (B, T, T)
print(wei[0])
# unlike the versions above, wei has different weights in different positions
# In the previous version, wei was initialized to a tensor with all 0's (T,T)

tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float("-inf"))
wei = F.softmax(wei, dim=1)
v = value(x)  # (B, T, headsize)
out = wei @ v  # (B, T, head_size)

print(wei[0])
