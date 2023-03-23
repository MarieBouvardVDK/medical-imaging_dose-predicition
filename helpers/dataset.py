import torch
import numpy as np
import os

class DoseDataset(torch.utils.data.Dataset):
    '''
    This class creates a DoseDataset to load the input data and real dose output.
    DoseDatasets will be used for the train and validation data.
    '''
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.samples = os.listdir(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.data_path + os.sep + self.samples[idx]
        
        #input data: ct + possible dose + structured masks
        ct_scan = torch.from_numpy(np.load(sample_path + os.sep + 'ct.npy')).float()
        possible_dose_mask = torch.from_numpy(np.load(sample_path + os.sep + 'possible_dose_mask.npy')).float()
        structure_masks = torch.from_numpy(np.load(sample_path + os.sep + 'structure_masks.npy')).float()
        
        #applying transform is required
        ct_scan = ct_scan.unsqueeze(0)
        if self.transform is not None:
            ct_scan = torch.cat([ct_scan, ct_scan, ct_scan], dim=0)
            ct_scan = self.transform(ct_scan)
            ct_scan = ct_scan[0].unsqueeze(0)

        #combine all input data
        concat_data = torch.cat((ct_scan, possible_dose_mask.unsqueeze(0), structure_masks), 0)
        
        #real dose 
        dose = torch.from_numpy(np.load(sample_path + os.sep + 'dose.npy'))
        
        return concat_data, dose.unsqueeze(0)
        
    


class TestDataset(torch.utils.data.Dataset):
    '''
    This class creates a TestDataset to load the input data and the corresponding file name.
    The TestDataset will be used for the test data.
    '''
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.samples = os.listdir(data_path)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.data_path + os.sep + self.samples[idx]

        #input data: ct scan + possible dose + structure masks
        ct_scan = torch.from_numpy(np.load(sample_path + os.sep + 'ct.npy')).float()
        possible_dose_mask = torch.from_numpy(np.load(sample_path + os.sep + 'possible_dose_mask.npy')).float()
        structure_masks = torch.from_numpy(np.load(sample_path + os.sep + 'structure_masks.npy')).float()
        
        #combine the input data
        concat_data = torch.cat((ct_scan.unsqueeze(0), possible_dose_mask.unsqueeze(0), structure_masks), 0)
        
        return concat_data, self.samples[idx]
