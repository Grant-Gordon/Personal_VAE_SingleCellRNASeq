import torch
import torch.nn as nn

class VaritationalAutoEncoder(nn.Module):

    def __init__(self, input_size):
        super(VaritationalAutoEncoder, self).__init__()

        #encoder
        self.encoder_fc1 = nn.Linear(input_size, 512)
        self.mu_layer = nn.Linear(512, 128)
        self.logvar_layer = nn.Linear(512, 128)
    
        #decoder
        self.decoder_fc1 = nn.Linear(128, 512)
        self.decoder_out = nn.Linear(512, input_size)
  
    
    def encode(self, x):
        h = self.relu(self.encoder_fc1(x))
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar
    
    def decode(self, z):
        h = self.relu(self.decoder_fc1(z))
        out = self.relu(self.decoder_out(h))
        return out



    #latent space sampling
    def reparameterize(self, mu, logvar):
        std =torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) #random noise from noraml dis
        return mu + eps * std # =z, returns latent vector sampled from 

    def forward(self,x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

       
    def vae_loss(reconstructed, original, mu, logvar):
        reconstructed_loss = nn.functional.mse_loss(reconstructed, original, reduction='sum') # or reduction = mean
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        total = reconstructed_loss + kl_div 
        return total, reconstructed_loss, kl_div