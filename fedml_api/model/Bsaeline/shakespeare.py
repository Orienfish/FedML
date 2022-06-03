import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import copy




SEQ_LEN = 80
NUM_CLASSES = 80
NUM_HIDDEN = 100

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.seq_len = SEQ_LEN  # window_size?
        self.num_classes = NUM_CLASSES  # vocab_size?
        self.embedding_dim = 8
        self.n_hidden = NUM_HIDDEN
        self.embedding = nn.Embedding(self.num_classes, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.n_hidden, batch_first=True)
        self.fc = nn.Linear(self.n_hidden, self.num_classes)

    def forward(self, x, prev_state):  # x: (n_samples, seq_len)
        emb = self.embedding(x)  # (n_samples, seq_len, embedding_dim)
        x, state = self.lstm(emb, prev_state)  # (n_samples, seq_len, n_hidden)
        logits = self.fc(x[:, -1, :])
        return logits, state

    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.n_hidden), torch.zeros(1, batch_size, self.n_hidden))
        
        

        
        
        
        
        
        
        
        
        
        
        
        
