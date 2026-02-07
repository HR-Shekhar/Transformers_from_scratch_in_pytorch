import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    """
    Docstring for InputEmbedding
    d_model: dimension of the embedding vector for each token
    vocab_size: size of the vocabulary of unique words 
    
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self):
        return self.embedding * math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))

        # Apply the sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # to add batch dimension => (1, seq_len, d_model)

        # registering it as a buffer
        # when you have a tensor that you want to keep it inside the model
        # but not as a trainable/learned parameter but you want it to be saved
        # when you save the file of the model, register it as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        # self.pe[batch_size(i.e. 1 here), take positions(seq_length(0 to seq_length-1)), d_model]
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    """
    Docstring for LayerNormalization
    In the paper PostLayerNorm is implemented modern architectures use PreLayerNorm
    *LayerNorm(x + Attention(x))*
    where x is the input embedding + Positional Encoding
    
    returns
    output shape: (B, T, d_model)
    """
    def __init__(self, eps: float = 10 ** -6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std =x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta
        
class FeedForward(nn.Module):
    """
    x is the output of LayerNorm of shape: (B, T, d_model) where B = 1   

    Feed forward in the paper
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    d_ff: inner layer dimensionality
    droput is added just for regularization
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)
        return x
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.h = h                  # no. of heads
        # to diviide d_model into vectors we need to ensure d_model % h == 0
        assert d_model % h == 0     # if d_model is not divisible by h, stop execution
        
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) #Wq
        self.w_k = nn.Linear(d_model, d_model) #Wk
        self.w_v = nn.Linear(d_model, d_model) #Wv

        self.w_o = nn.Linear (d_model, d_model) #Wo : The final output matrix(all heads concat)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        key = self.w_k(k)    # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        value = self.w_v(v)  # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)

        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k)
        