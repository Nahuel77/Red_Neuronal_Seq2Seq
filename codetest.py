import torch
import torch.nn as nn
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = 10         # Tokens posibles (0..9)
seq_length = 5          # Longitud fija de las secuencias
embedding_dim = 16
hidden_dim = 32
num_epochs = 2000
batch_size = 64

cad = torch.tensor([[1,2,3],[4,5,9],[6,7,8]])

print(cad)

print(cad.unsqueeze(1))

print(cad.squeeze(1))