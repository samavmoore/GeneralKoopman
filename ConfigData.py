
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule

class PendulumKoopmanDataset(Dataset):

    def __init__(self, n_states: int=2,
                    n_shifts: int=32,
                    n_traj: int=5000,
                    traj_len: int=51,
                    data_path: str='my/path'):
    
        super().__init__()
        # load orig_data
        orig_data = np.load(data_path)

        # shift the data
        shifted_data = shift_n_stack(orig_data, n_shifts, n_traj, traj_len)

        self.states = shifted_data[0,0:n_states,:]
        self.context = shifted_data[0,n_states:,:]
        self.future_states = shifted_data[1:,0:n_states,:]
        self.n_samples = len(shifted_data[0,0,:])

    def __getitem__(self, index):
        return self.states[:,index], self.context[:,index], self.future_states[:,:, index]
        
    def __len__(self):
        return self.n_samples


def shift_n_stack(data, n_shifts, n_traj, traj_len):
    '''
        Purpose: Shift and stack the original data so that the state values at a later time (up to n_shifts later) 
                are easily accessible during training for loss calculation.
    '''

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


class PendulumKoopmanDataModule(LightningDataModule):
    def __init__(self, batch_size: int=64,
                    n_states: int=2,
                    n_shifts: int=32,
                    traj_len: int=51,
                    train_n_traj: int=5000,
                    val_n_traj: int=20, 
                    train_data_path: str='my/path',
                    val_data_path: str='my/path'):
        super().__init__()

        self.n_states = n_states
        self.n_shifts = n_shifts
        self.traj_len = traj_len
        self.train_n_traj = train_n_traj
        self.val_n_traj = val_n_traj
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.batch_size = batch_size

    def setup(self, stage: 'str'):
        if stage == "fit":

            self.koop_train = PendulumKoopmanDataset(n_shifts=self.n_shifts, n_states=self.n_states, \
            n_traj=self.train_n_traj, traj_len= self.traj_len, data_path=self.train_data_path)

            self.koop_val = PendulumKoopmanDataset(n_shifts=self.n_shifts, n_states=self.n_states, \
            n_traj=self.val_n_traj, traj_len= self.traj_len, data_path=self.val_data_path)

    def train_dataloader(self):
        return DataLoader(self.koop_train, batch_size=self.batch_size, shuffle=True)
        
    def val_dataloader(self):
        return DataLoader(self.koop_val, batch_size=self.batch_size)


class ContextDataset(Dataset):
    def __init__(self, data_path: str="my/path"):
        super().__init__()
        self.data = np.load(data_path)

        self.data = torch.from_numpy(self.data).float()

    def __getitem__(self, index):
        return self.data[index,:]

    def __len__(self):
        return len(self.data[:,0])


class ContextDataModule(LightningDataModule):
    def __init__(self, data_path: str='my/path',
                    batch_size: int=64):
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size

    def setup(self, stage: str):
        if stage == "fit":
            self.full_data = ContextDataset(self.data_path)
            self.train, self.val = random_split(self.full_data, [700, 300])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)
        
    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

class EigenPretrainDataset(Dataset):
    def __init__(self, data_path: str='my/path',
                    n_states: int=2):
        super().__init__()
        self.data = np.load(data_path)
        self.data = torch.from_numpy(self.data).float()

        self.states = self.data[:,:n_states]
        self.context = self.data[:,n_states:]

    def __getitem__(self, index):
        return self.states[index,:], self.context[index,:]

    def __len__(self):
        return len(self.data[:,0])

class EigenPretrainModule(LightningDataModule):
    def __init__(self, data_path: str='my/path',
                    batch_size: int=128):
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size

    def setup(self, stage: str):
        if stage == 'fit':
            self.full_data = EigenPretrainDataset(data_path=self.data_path)
            self.train, self.val = random_split(self.full_data, [178500,76500])

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)
