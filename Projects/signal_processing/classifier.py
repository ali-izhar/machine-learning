from torch import nn, optim
import pytorch_lightning as pl
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, input_shape=(1, 1, 22016)):
        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_shape[1], out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_size, num_classes=10):
        super(LinearBlock, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class AudioClassifier(pl.LightningModule):
    def __init__(self, lr, input_shape=(1, 1, 22016), num_classes=10):
        super(AudioClassifier, self).__init__()
        self.lr = lr
        self.conv = ConvBlock(input_shape)
        flattened_size = 64 * (input_shape[2] // 8)
        self.linear = LinearBlock(flattened_size, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.linear(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.nll_loss(y_hat, y)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)
