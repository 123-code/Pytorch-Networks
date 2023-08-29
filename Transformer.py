# following yt tutorial

import torch 
import torch.nn as nn
import math

# mapping of letters and vectors of size 512
class InputEmbedding(nn.Module):
  
  def __init__(self,d_model:int,vocab_size:int):
    super().__init__()
    self.d_model = d_model
    self.vocab_size = vocab_size
    self.embedding = nn.Embedding(vocab_size,d_model)

  def forward(self,x):
    return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
  def __init__(self,d_model:int,seq_len:int,dropout:float) -> None:
    super().__init__()
    self.d_model = d_model
    self.seq_len = seq_len
    self.dropout = nn.Dropout(dropout)

    positionalencoding = torch.zeros(seq_len,d_model)
    # vector that represents position of the word inside the sentence
    position = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
    # denominator of encoding function
    denominator = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
    # sin to even positions
    # all wil have the sine, starting from zero, towards the end and going forward by 2. 
    positionalencoding[:,0::2] = torch.sin(position * denominator)
    positionalencoding[:,1::2] = torch.cos(position * denominator)

    positionalencoding = positionalencoding.unsqueeze(0)

    #register tensor in the buffer of module, meaning it will be saved on the file, with the model state.

 
    self.register_buffer('pe',positionalencoding)

  def forward(self,x):

    x = x + (self.pe[:,:x.shape[1],:]).requires_grad(False)
    return self.dropout(x)
