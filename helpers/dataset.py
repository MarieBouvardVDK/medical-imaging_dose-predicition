import torch
import numpy as np
import os

class DoseDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, test=False, transform=None):
        self.data_path = data_path
        self.samples = os.listdir(data_path)
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.data_path + os.sep + self.samples[idx]

        ct_scan = torch.from_numpy(np.load(sample_path + os.sep + 'ct.npy')).float()
        possible_dose_mask = torch.from_numpy(np.load(sample_path + os.sep + 'possible_dose_mask.npy')).float()
        structure_masks = torch.from_numpy(np.load(sample_path + os.sep + 'structure_masks.npy')).float()
      
        concat_data = torch.cat((ct_scan.unsqueeze(0), possible_dose_mask.unsqueeze(0), structure_masks), 0)

        if self.test == False:
            dose = torch.from_numpy(np.load(sample_path + os.sep + 'dose.npy'))
            return concat_data, dose.unsqueeze(0)
        
        return concat_data
