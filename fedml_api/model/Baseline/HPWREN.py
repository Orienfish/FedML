import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import copy


SEQ_LEN = 24
NUM_INPUT = 12  # input_size, corresponds to the number of features in the input
NUM_OUTPUT = 12  # output_size, the number of items in the output
NUM_HIDDEN = 128
NUM_LAYERS = 1

class HPWREN_Net(nn.Module):

    def __init__(self):
        super(HPWREN_Net, self).__init__()

        self.num_layers = NUM_LAYERS

        self.input_size = NUM_INPUT
        self.hidden_size = NUM_HIDDEN
        self.output_size = NUM_OUTPUT

        self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x, prev_state):
        #h_0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(device)
        #c_0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(device)

        out, (h_out, c_out) = self.lstm(x, prev_state)  # b, t, output_size

        out = self.fc(out)
        out = out[:, -1, :]

        return out
















