import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import copy

FEATURE_DIM = 561
NUM_CLASSES = 6

class HAR_Net(nn.Module):
    def __init__(self):
        super(HAR_Net, self).__init__()
        self.fc1 = nn.Linear(FEATURE_DIM, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
















