import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import math

class DeepAutoEncoder(nn.Module):

    def __init__(self, input_size):
        super(DeepAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU()
        )
       
        self.decoder = nn.Sequential(
           nn.Linear(128,512),
           nn.ReLU(),
           nn.Linear(512,input_size)
        )


    def forward(self,x):
        encoded = self.encoder(x)
        decoded= self.decover(encoded)
        return decoded

