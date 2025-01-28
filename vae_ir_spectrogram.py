#variational autoencoder for impulse responses with spectrogram input
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
import json

# model hyperparameters
dataset_path = '../data/spectrograms.npy'
labels_path = '../data/label_list.npy'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 2
learning_rate = 1e-5
num_epochs = 100

# load dataset
dataset = np.load(dataset_path, allow_pickle=True)
labels = np.load(labels_path, allow_pickle=True)
num_samples = dataset.shape[0]
input_size0 = dataset.shape[1]
input_size1 = dataset.shape[2]

# define model params

hidden_size = 512
latent_size = 32
output_size = input_size0 * input_size1

# define model

class VAE_gc(nn.Module):
    def __init__(self, input_size0, input_size1, hidden_size, latent_size, output_size):
        super(VAE_gc, self).__init__()
        self.fc1 = nn.Linear(input_size0 * input_size1, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, input_size0 * input_size1))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# instantiate model
model = VAE_gc(input_size0, input_size1, hidden_size, latent_size, output_size).to(DEVICE)

# define loss function
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_size0 * input_size1), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# define optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(num_epochs):
    for i in range(num_samples):
        # get data
        data = dataset[i]
        data = torch.tensor(data).float().to(DEVICE)
        data = data.view(-1, input_size0 * input_size1)

        # forward pass
        recon_data, mu, logvar = model(data)

        # compute loss
        # print(epoch, i)
        # print(torch.min(recon_data))
        # print(torch.max(data))

        loss = loss_function(recon_data, data, mu, logvar)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, num_samples, loss.item()))

        # print loss
        if i % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, num_samples, loss.item()))

# save model
torch.save(model.state_dict(), 'vae_ir_spectrogram.pth')
print('Model saved')

# save model params
model_params = {
    'input_size0': input_size0,
    'input_size1': input_size1,
    'hidden_size': hidden_size,
    'latent_size': latent_size,
    'output_size': output_size
}
with open('vae_ir_spectrogram.json', 'w') as f:
    json.dump(model_params, f)

print('Model params saved')

# save model hyperparams
model_hyperparams = {
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'num_epochs': num_epochs
}
with open('vae_ir_spectrogram_hyperparams.json', 'w') as f:
    json.dump(model_hyperparams, f)

print('Model hyperparams saved')

# save model training data

model_training_data = {
    'loss': loss.item()
}
with open('vae_ir_spectrogram_training_data.json', 'w') as f:
    json.dump(model_training_data, f)

print('Model training data saved')

# generate some random data
z = torch.randn(16, latent_size).to(DEVICE)
recon_data = model.decode(z)
recon_data = recon_data.view(input_size0, input_size1).cpu().detach().numpy()
np.save('recon_data.npy', recon_data)
print('Reconstructed data saved')




