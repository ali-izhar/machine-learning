import logging
import torch
import torch.nn.functional as F
from torch import optim
from pytorch_lightning import LightningModule

from .autoencoder import AutoEncoder
from .codebook import VQCodebook


class VQAutoEncoder(LightningModule):
    def __init__(self, learning_rate, normalize=True, codebook_size=512, autoencoder=None):
        super(VQAutoEncoder, self).__init__()

        if autoencoder is None:
            autoencoder = AutoEncoder(learning_rate, normalize)

        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder
        self.codebook_dim = autoencoder.encoder.codebook_dim
        
        self.codebook = VQCodebook(self.codebook_dim, codebook_size)
        self.learning_rate = learning_rate
        self.normalize = normalize

    def forward(self, x):
        logging.info(f"Expected input shape: (batch_size, 1, sequence_length)")
        logging.info(f"Input shape: {x.shape}")

        if self.normalize:
            x, x_min, x_max = self._normalize(x)

        z_e = self.encoder(x)
        logging.info(f"Encoder output shape: {z_e.shape}")

        z_q, _, vq_loss = self.codebook(z_e)
        logging.info(f"Codebook output shape: {z_q.shape}")

        x_hat = self.decoder(z_q)
        logging.info(f"Decoder output shape: {x_hat.shape}")

        loss = self._reconstruction_loss(x, x_hat) + vq_loss
        if self.normalize:
            x_hat = self._denormalize(x_hat, x_min, x_max)

        return x_hat, loss

    def _reconstruction_loss(self, original, reconstruction):
        return F.mse_loss(reconstruction, original, reduction='mean')
        
    def training_step(self, batch, batch_idx):
        x, _ = batch
        _, loss = self.forward(x)
        self.log('loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        _, loss = self.forward(x)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)

    @torch.no_grad()
    def _normalize(self, x):
        x_min = x.min(dim=-1, keepdim=True)[0]
        x_max = x.max(dim=-1, keepdim=True)[0]
        return ((x - x_min) / (x_max - x_min)), x_min, x_max

    @torch.no_grad()
    def _denormalize(self, x_norm, x_min, x_max):
        return x_norm * (x_max - x_min) + x_min
