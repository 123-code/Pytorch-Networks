import torch 
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self,d_model:int,vocab_size:int):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):            # !!
        return self.embedding(x) * math.sqrt(self.d_model)
        
        
class PositionalEncoding(nn.Module):
  def __init__(self,d_model:int,seq_len:int,dropout:float)->None:
      super().__init__()
      self.d_model = d_model
      self.seq_len = seq_len
      self.dropout = nn.Dropout(dropout)

      '''
      matriz de forma seq_len,d_model, siendo seq_len tamaño de la frase\n
      y d_model tamaño del vector representando a cada palabra.

      muchos vectores de longitud 512, el numero\n
      de vectores es igual a la cantidad de palabras en la frase
      '''

      matriz = torch.zeros(seq_len,d_model) 
      # vector representa la posicion de la palabra en la frase.
      pos = torch.arange(0,seq_len,dtype=torch.float).unsqueeze(1)
      #!! vector del denominador en positional encoding
      div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))

      '''
      en la matriz que calculamos, a las posiciones pares\n
      se les saca el seno de su posicion multiplicado por el denominador\n
      a las posiciones impares se les saca el coseno de su posicion multiplicado por el denominador
      '''
      
      matriz[:,0::2] = torch.sin(pos*div_term)
      matriz[:,1::2]= torch.cos(pos*div_term)

      matriz = matriz.unsqueeze_(0)
      # guardar la matriz en el buffer 
      self.register_buffer('matriz',matriz)
  def forward(self,x):
      x = x + (self.pos[:,x.shape[1],:]).requires_grad_(False)
      return self.dropout(x)
  
class LayerNormalization(nn.Module):
    def __init__(self,eps:float=10**-6)->None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x):
        mean = x.mean(dim=-1,keepdim=True)


      

 


    
     
    
