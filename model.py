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
        
class FeedForwardBlock(nn.Module):
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

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (Batch, h, Seq_len, d_model, d_k) --> (Batch, h, Seq_len, Seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill_(mask == 0 , -1e9) 
        attention_scores = attention_scores.softmax(dim = -1) # (Batch, h, seq_len, seq_len)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)  # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        key = self.w_k(k)    # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        value = self.w_v(v)  # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)

        # (Batch, Seq_len, d_model) --> (Batch, Seq_lenh, h, d_k) --> (Batch, h, Seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, Seq_len, h, d_k) --> (Batch, Seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, Seq_len, d_model) --> (Batch, Seq_len, d_model)
        return self.w_o(x)
        
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.Module([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):  
        """
        src_mask in a Transformer encoder block primarily prevents valid (regular) tokens 
        from attending to padding tokens during self-attention.
        """
        
        # self_attention_block(query, key, value) because x is the query, key and value itself
        # that's why it is called self attention block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, mask):
        for layers in self.layers:
            x = layers(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for i in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        
        """
        tgt_mask Prevents the decoder from "peeking" at future tokens in the target (output) sequence during self-attention

        src_mask Masks padding tokens in the source (encoder) sequence during cross-attention
        """
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, lambda x: self.feed_forward_block)
        return x
    
class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
    

class ProjectionLAyer(nn.Module):
    """
    This implements the final linear layer and softmax layer that comes after the decoder layer.
    Its called projection layer because its projecting the embedding into the vocabulary.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, Seq_Len, d_model) --> (Batch, Seq_len, Vocab_Size)
        return torch.log_softmax(self.proj(x), dim = -1)
    
# class Transformer(nn.Module):
#     def __init__(self, encoder:Encoder, decoder: Decoder):e
#         super().__init__()