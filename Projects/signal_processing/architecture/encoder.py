from torch import nn
from itertools import chain


class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_sizes, codebook_dim):
        super(Encoder, self).__init__()
        
        layers = []
        layer_sizes = list(chain([input_channels], hidden_sizes, [codebook_dim]))
        
        for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Conv1d(input_size, output_size, 3, padding = 1, stride = 2))
            layers.append(nn.InstanceNorm1d(output_size))
            layers.append(nn.ReLU())
        
        self.encoder = nn.Sequential(*layers)
        self.codebook_dim = codebook_dim

    def forward(self, x):
        z_e = self.encoder(x)
        z_e = z_e.view(-1, self.codebook_dim, 1) # add dimension for codebook
        return z_e
