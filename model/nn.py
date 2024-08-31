import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
  def __init__(self, d_model: int, vocab_size: int) -> None:
    super().__init__()
    self.d_model = d_model                              #embedding vector dim
    self.vocab_size = vocab_size                        #input dictionary size
    self.embedding = nn.Embedding(vocab_size, d_model)  #construct a lookup table that stores embeddings of a fixed dictionary and size
  
  def forward(self, x):
    # get embedding weights and multiply those weights by sqrt(d_model)
    return self.embedding(x) * math.sqrt(self.d_model)  #output shape: (batch, seq_len, embedding_dim[d_model])

class PositionalEncoding(nn.Module):
  def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
    super().__init__()
    self.d_model = d_model
    self.seq_len = seq_len
    self.dropout = nn.Dropout(dropout)
    # create a matrix to store positional encoding with shape (seq_len, d_model)
    pe = torch.zeros(seq_len, d_model)
    # create a vector to represent position of word in sequence with shape (seq_len, 1)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    # create the denominator term in log format for math stability
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # this is equivalent to (1/10000)^(2i/d_model)
		# apply sine to even indices
    pe[:, 0::2] = torch.sin(position * div_term)
    # apply cosine to odd indices
    pe[:, 1::2] = torch.cos(position * div_term)
    # add a batch dimension to the positional encoding
    pe = pe.unsqueeze(0)
    # register the positional encoding as a buffer
    self.register_buffer('positional_encoding', pe)
    
  def forward(self, x):
    # add positional encoding for every word in the sequence
    # we do not need to learn the position encoding
    x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
    return self.dropout(x)
  

class LayerNormalization(nn.Module):
  def __init__(self, features: int, eps: float=10**-6) -> None:
    super().__init__()
    self.eps = eps
    self.alpha = nn.Parameter(torch.ones(features))
    self.bias = nn.Parameter(torch.ones(features))
  
  def forward(self, x):
    mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
    std = x.std(dim = -1, keepdim = True)   # (batch, seq_len, 1)
    return self.alpha * (x - mean) / (std + self.eps) + self.bias
  

class FeedForward(nn.Module):
  def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
    super().__init__()
    self.linear_1 = nn.Linear(d_model, d_ff)  # w_1, b_1
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(d_ff, d_model)  # w_2, b_2
    
  def forward(self, x):
    return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
  
  
class ResidualConnection(nn.Module):
  def __init__(self, features: int, dropout: float) -> None:
    super().__init__()
    self.dropout = nn.Dropout(dropout)
    self.layernor = LayerNormalization(features)
  
  # [reference to paper]: we employ a residual connection around each of the two sub-layers, followed
  # by layer normalization. That is, the output of each sub-layer is LayerNorm(x + sublayer(x))
  # where sublayer(x) is the function implemented by the sub-layer itself
  def forward(self, x, sublayer):
    return self.norm(x + self.dropout(sublayer(x)))


class MultiHeadAttention(nn.Module):
  def __init__(self, d_model: int, h: int, dropout: float) -> None:
    super().__init__()
    self.d_model = d_model     # embedding dim
    self.h = h                 # number of heads
    assert d_model % h == 0, "d_model is not divisible by h"
    
    
    '''
    [paper reference]: MultiHead(Q, K, V) = Concat(head1,...,headh)Wo
                       where headi = Attention(QW_i^Q, KW_i^K, VW_i^V)
    '''
    self.d_k = d_model // h    # dimension of vector seen by each head
    self.w_q = nn.Linear(d_model, d_model, bias=False)
    self.w_k = nn.Linear(d_model, d_model, bias=False)
    self.w_v = nn.Linear(d_model, d_model, bias=False)
    self.w_o = nn.Linear(d_model, d_model, bias=False)
    self.dropout = nn.Dropout(dropout)
    
  
  @staticmethod
  def attention(query, key, value, mask, dropout: nn.Dropout):
    # Attention(Q, K, V) = softmax(QK^T/sqrt(d_k))V
    d_k = query.shape[-1]
    attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
      # apply mask by setting positions where mask == 0 to -inf
      attention_scores.masked_fill(mask == 0, -1e9)
    attention_scores = attention_scores.softmax(dim=-1)
    if dropout is not None:
      attention_scores = dropout(attention_scores)
    attention_scores = attention_scores @ value
    # final shape: (batch, h, seq_len, d_k)
    return attention_scores

  def forward(self, q, k, v, mask):
    query = self.w_q(q)
    key = self.w_k(k)
    value = self.w_v(v)
    
    # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
    query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
    key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
    value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
    
    # calculate attention
    attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
    
    # combine all heads together
    # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
    attention_scores = attention_scores.transpose(1, 2).contiguous().view(attention_scores.shape[0], attention_scores.shape[1], self.d_k * self.h )
    
    # multiply by Wo
    return self.w_o(attention_scores)
