import xarray as xr
import numpy as np
from torch.utils.data import Dataset
import torch


def load_nc_data(root):
    """load nc data and select the region of interest
    
    Returns:
        data: a xr.dataarray of shape (1590, 64, 64)
    """
    data_file = root + 'saved_aod_2023_interp.nc'
    var_name = 'aod'
    with xr.open_dataset(data_file) as data:
        data = data[var_name]
    return data


def split_data(data, is_train):
    """split data into training, validation and testing sets

    Args:
        data: a xr.dataarray of shape (1590, 64, 64)
        is_train: a boolean indicating whether to split training set or not

    Returns:
        One of the following:
        train_data: a numpy array
        valid_data: a numpy array
        test_data: a numpy array
    """
    if is_train:
        train_data = data.values[:9600,:,:]
        return train_data
    else:
        valid_data = data.values[9600:12000,:,:]
        return valid_data
    # test_data = data.values[12000:13200,:,:]


class NcDataset(Dataset):
    def __init__(self, is_train, root, n_frames_input, n_frames_output):
        super().__init__()
        self.datas = load_nc_data(root)
        self.num_frames_input = n_frames_input
        self.num_frames_output = n_frames_output
        self.num_frames = n_frames_input + n_frames_output
        self.datas = split_data(self.datas, is_train)
        print('Loaded {} samples ({})'.format(self.__len__(), 'train' if is_train else 'valid'))

    def __getitem__(self, idx):
#        data = self.datas[index:index+self.num_frames]
        data = self.datas[idx*self.num_frames:(idx+1)*self.num_frames]
        inputs = data[:self.num_frames_input]
        targets = data[self.num_frames_input:]
        inputs = inputs[..., np.newaxis]
        targets = targets[..., np.newaxis]
        # replace nan values with 0
        # inputs = np.nan_to_num(inputs)
        # targets = np.nan_to_num(targets)
        inputs = torch.from_numpy(inputs).permute(0, 3, 1, 2).float().contiguous()
        targets = torch.from_numpy(targets).permute(0, 3, 1, 2).float().contiguous()
        return idx, targets, inputs
    

    def __len__(self):
        return self.datas.shape[0] // self.num_frames


if __name__ == "__main__":
    trainFolder = NcDataset(is_train=True,
                            root='./',
                            n_frames_input=6,
                            n_frames_output=6)
    validFolder = NcDataset(is_train=False,
                            root='./',
                            n_frames_input=6,
                            n_frames_output=6)
    trainLoader = torch.utils.data.DataLoader(trainFolder,
                                                batch_size=8,
                                                shuffle=False)
    validLoader = torch.utils.data.DataLoader(validFolder,
                                                batch_size=8,
                                                shuffle=False)
    for i, (idx, output, input) in enumerate(trainLoader):
        print(i)
        print(idx)
        print(output.size())
        print(input.size())
        break
