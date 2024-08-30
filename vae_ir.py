# variational autoencoder for impulse responses

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import json

# model hyperparameters
dataset_path = 'data/labeled_wav_data_resampled.npy'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2
learning_rate = 1e-5
num_epochs = 100

# load dataset
dataset = np.load(dataset_path, allow_pickle=True)
# unpack dict into labels and data
labels = dataset.item().keys()
data = np.array([dataset.item().get(label) for label in labels])
num_samples = data.shape[0]
input_size = data.shape[1]


# define model params
input_size = data.shape[1]
hidden_size = 4096
latent_size = 256
output_size = data.shape[1]

# define model
# this block written by copilot....
# class VAE_gc(nn.Module):
#     def __init__(self, input_size, hidden_size, latent_size, output_size):
#         super(VAE_gc, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc21 = nn.Linear(hidden_size, latent_size)
#         self.fc22 = nn.Linear(hidden_size, latent_size)
#         self.fc3 = nn.Linear(latent_size, hidden_size)
#         self.fc4 = nn.Linear(hidden_size, output_size)

#     def encode(self, x):
#         h1 = F.relu(self.fc1(x))
#         return self.fc21(h1), self.fc22(h1)

#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5*logvar)
#         eps = torch.randn_like(std)
#         return mu + eps*std

#     def decode(self, z):
#         h3 = F.relu(self.fc3(z))
#         return torch.sigmoid(self.fc4(h3))

#     def forward(self, x):
#         mu, logvar = self.encode(x.view(-1, input_size))
#         z = self.reparameterize(mu, logvar)
#         return self.decode(z), mu, logvar
class Encoder(nn.Module):
    
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat
    
class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)        # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)
        
        return x_hat, mean, log_var
    
# instantiate model
# model = VAE_gc(input_size, hidden_size, latent_size, output_size)
encoder = Encoder(input_dim=input_size, hidden_dim=hidden_size, latent_dim=latent_size)
decoder = Decoder(latent_dim=latent_size, hidden_dim = hidden_size, output_dim = input_size)

model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(num_epochs):
    for i in range(0, num_samples, batch_size):
        # get batch
        batch = data[i:i+batch_size]
        batch = torch.tensor(batch, dtype=torch.float32)
        
        # forward pass
        output, mu, logvar = model(batch)
        #count nan values
        print(torch.isnan(output).sum())
        print(torch.min(output))
        print(torch.min(batch))
        # loss
        reconstruction_loss = F.binary_cross_entropy(output, batch, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = reconstruction_loss + kl_div
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print loss
        if i % 1000 == 0:
            print(f'Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}')
