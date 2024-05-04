from torch import nn
from itertools import chain


class Decoder(nn.Module):
    def __init__(self, codebook_dim, hidden_sizes, output_channels):
        super(Decoder, self).__init__()

        layers = []

        # NOTE: Do not reverse the hidden sizes as they were already reversed in the AutoEncoder class
        layer_sizes = list(chain([codebook_dim], hidden_sizes, [output_channels]))
        
        for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.ConvTranspose1d(input_size, output_size, 3, padding=1, stride=2, output_padding=1))
            layers.append(nn.InstanceNorm1d(output_size))
            layers.append(nn.ReLU())
        
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.decoder(x)
        return x
