import torch
import torch.nn as nn

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
        decoded= self.decoder(encoded)
        return decoded

