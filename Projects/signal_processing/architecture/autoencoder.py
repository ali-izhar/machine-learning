import torch
import torch.nn.functional as F
from torch import optim
from pytorch_lightning import LightningModule

from .encoder import Encoder
from .decoder import Decoder


class AutoEncoder(LightningModule):
    def __init__(self, learning_rate, normalize=True):
        super(AutoEncoder, self).__init__()

        # NOTE: The hidden sizes are reversed. Do not reverse them in the Decoder class
        self.encoder = Encoder(1, [512, 256, 128, 64], 32)
        self.decoder = Decoder(32, [64, 128, 256, 512], 1)
        self.learning_rate = learning_rate
        self.normalize = normalize

    def forward(self, x):
        if self.normalize:
            x, x_min, x_max = self._normalize(x)
        
        z_e = self.encoder(x)
        reconstruction = self.decoder(z_e)
        loss = self._reconstruction_loss(x, reconstruction)
        if self.normalize:
            reconstruction = self._denormalize(reconstruction, x_min, x_max)
        return reconstruction, loss

    def _reconstruction_loss(self, original, reconstruction):
        return F.mse_loss(reconstruction, original, reduction='mean')

    def training_step(self, batch, batch_index):
        x, _ = batch
        _, loss = self.forward(x)
        self.log('recon_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)
        
    @torch.no_grad()
    def _normalize(self, x):
        x_min = x.min(dim=2, keepdim=True)[0]
        x_max = x.max(dim=2, keepdim=True)[0]
        return (x - x_min) / (x_max - x_min), x_min, x_max

    @torch.no_grad()
    def _denormalize(self, x_norm, x_min, x_max):
        return x_norm * (x_max - x_min) + x_min
