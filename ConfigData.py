
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class CustomKoopmanDataset(Dataset):

    def __init__(self, parameters, data_name):
        # load orig_data
        orig_data = np.load(data_name)

        # shift the data
        shifted_data = shift_n_stack(orig_data, parameters)

        n_states = parameters['n_states']

        self.states = shifted_data[0,0:n_states,:]
        self.context = shifted_data[0,n_states:,:]
        self.future_states = shifted_data[1:,0:n_states,:]
        self.n_samples = len(shifted_data[0,0,:])

    def __getitem__(self, index):
        return self.states[:,index], self.context[:,index], self.future_states[:,:, index]
        
    def __len__(self):
        return self.n_samples


def shift_n_stack(data, params):
    '''
        Purpose: Shift and stack the original data so that the state values at a later time (up to n_shifts later) 
                are easily accessible during training.
    '''

    n_shifts = params['n_shifts']
    traj_len = params['traj_len']
    n_traj = params['n_traj']
    data = data.astype('float32')
    dim0, dim1 = data.shape
    new_dim = (traj_len-n_shifts)*n_traj
    stacked_data = np.zeros((n_shifts+1, dim1, new_dim))
    
    new_traj_count = 0 
    for i in range(n_traj):
        # grab each trajectory
        traj_start = i*traj_len
        traj_end = (i+1)*traj_len
        traj = data[traj_start:traj_end, : ]
        #print(f'stacking data: traj {i}')

        counter = 0
        while (counter + traj_start + n_shifts) < traj_end:
            #shift the sliding window and stack in a 3d array
            shifted_traj = traj[counter: counter+ n_shifts+1, :]
            stacked_data[:,:,new_traj_count] = shifted_traj
            counter += 1
            new_traj_count += 1

    stacked_data = torch.from_numpy(stacked_data).float()
    return stacked_data


class KoopmanDataModule(LightningDataModule):
    def __init__(self, train_dict, val_dict, train_data_name, val_data_name):
        super().__init__()
        self.train_dict = train_dict
        self.val_dict = val_dict
        self.train_data_name = train_data_name
        self.val_data_name = val_data_name
        self.batch_size =train_dict['batch_size']

    def setup(self, stage: 'str'):
        if stage == "fit":
            self.koop_train = CustomKoopmanDataset(self.train_dict, self.train_data_name)
            self.koop_val = CustomKoopmanDataset(self.val_dict, self.val_data_name)

    def train_dataloader(self):
        return DataLoader(self.koop_train, batch_size=self.batch_size, shuffle=True)
        
    def val_dataloader(self):
        return DataLoader(self.koop_val, batch_size=self.batch_size)


