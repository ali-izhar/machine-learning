import os
from glob import glob
import librosa
from torch.utils.data import Dataset
import torch
import numpy as np

def convert_np_to_torch(numpy_data):
    return torch.Tensor(np.array(numpy_data, dtype=np.float32))

class AudioMNIST(Dataset):
    def __init__(self, root, target_sample_rate=22050, normalize=True, duration=None, transform=None):
        super(AudioMNIST, self).__init__()
        self.root = root
        self.target_sample_rate = target_sample_rate
        self.normalize = normalize
        self.duration = duration
        self.transform = transform
        self.wav_files = glob('*/*.wav', root_dir=root, recursive=True)
        self.min_max = {}

    def __getitem__(self, index):
        file_name = os.path.join(self.root, self.wav_files[index])
        label = int(os.path.basename(file_name).split('_')[0])
        waveform, _ = librosa.load(file_name, sr=self.target_sample_rate, duration=self.duration)

        if self.normalize:
            if label not in self.min_max:
                self.min_max[label] = (np.min(waveform), np.max(waveform))
            min_val, max_val = self.min_max[label]
            waveform = (waveform - min_val) / (max_val - min_val)
        
        if self.transform is not None:
            waveform = self.transform(waveform)
        
        waveform = convert_np_to_torch(waveform).unsqueeze(0)
        return waveform, label

    def __len__(self):
        return len(self.wav_files)
