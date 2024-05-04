import torch
from torch import nn

class VQCodebook(nn.Module):
    def __init__(self, codebook_dim, num_codewords, beta=0.25):
        super(VQCodebook, self).__init__()
        
        self.embedding = nn.Embedding(num_codewords, codebook_dim)
        self.embedding.weight.data.uniform_(-1, 1)
        self.codebook_dim = codebook_dim
        self.num_codewords = num_codewords
        self.beta = beta

    def forward(self, z_e):
        batch_size, channels, length = z_e.shape

        if channels != self.codebook_dim:
            raise RuntimeError(f'Expected input to have {self.codebook_dim} channels, got {channels}')

        z_e = z_e.permute(2, 1, 0).contiguous()
        z_e_flat = z_e.view(-1, self.codebook_dim)

        d = torch.sum(z_e_flat ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_e_flat, self.embedding.weight.t())
        
        indices = torch.argmin(d, dim=1)
        quantized = self.embedding(indices).view(z_e.shape)

        vq_loss = torch.mean((quantized.detach() - z_e) ** 2) + \
                  self.beta * torch.mean((quantized - z_e.detach()) ** 2)

        return quantized, indices, vq_loss
