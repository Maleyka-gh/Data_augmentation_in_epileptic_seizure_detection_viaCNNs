# Generator synthetic Mond Data
# Made them to a Pytorch Dataset

from torch.utils.data import Dataset, DataLoader
import torch
from GANModels import *
import numpy as np
import os


class Synthetic_Dataset(Dataset):
    def __init__(self,
                 mond_model_path=r'C:\Users\seyidova\Desktop\tts_gan\checkpoint_bestmse',
                 sample_size=300
                 ):
        self.sample_size = sample_size

        # Generate seizure Data
        mond_gen_net = Generator(seq_len=500, channels=3, latent_dim=100, embed_dim=10, patch_size=50)
        mond_ckp = torch.load(mond_model_path, map_location='cpu')
        mond_gen_net.load_state_dict(mond_ckp['gen_state_dict'])

        # generate synthetic seizure data - label is 1
        z = torch.FloatTensor(np.random.normal(0, 1, (self.sample_size, 100)))  # latent_dim=100
        self.syn_seiz = mond_gen_net(z)
        self.syn_seiz = self.syn_seiz.detach().numpy()
        self.syn_label = np.ones(len(self.syn_seiz))
        self.syn_label = self.syn_label.reshape(self.syn_label.shape[0], 1)

        print(self.syn_seiz.shape)
        print(self.syn_label.shape)

    def __call__(self):
        return self.syn_seiz

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        return self.syn_seiz[idx], self.syn_label[idx]


