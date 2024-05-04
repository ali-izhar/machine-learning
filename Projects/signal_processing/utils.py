import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T

import custom_transforms as CT
from AudioMNIST import AudioMNIST

def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_model(model, save_path):
    model_scripted = model.to_torchscript(method="trace")
    model_scripted.save(save_path)

def load_model(file_path):
    model = torch.jit.load(file_path)
    return model.eval()

def get_dataloaders(root='./AudioMNIST/data', normalize=False, sample_rate=22050, n_fft=512, hop_length=256, n_mels=64, SAMPLE_RND=22016):
    dataset = AudioMNIST(
        root,
        target_sample_rate=sample_rate,
        transform=T.Compose([
            # CT.TrimSilence(15),
            CT.FixLength(SAMPLE_RND),
            CT.FFT(),
        ]),
        normalize=normalize
    )

    total_dataset_length = len(dataset)
    validation_dataset_length = int(total_dataset_length * 0.1)

    train_dataset_length = total_dataset_length - validation_dataset_length
    train_dataset, validation_dataset = random_split(dataset, [train_dataset_length, validation_dataset_length])

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )
    
    return train_loader, validation_loader