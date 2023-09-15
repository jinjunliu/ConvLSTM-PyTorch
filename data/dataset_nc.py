import xarray as xr
import numpy as np
from torch.utils.data import Dataset
import torch


def load_nc_data(config):
    """load nc data and select the region of interest
    
    Returns:
        data: a xr.dataarray of shape (1590, 64, 64)
    """
    data_file = config.data_file
    var_name = config.var_name
    with xr.open_dataset(data_file) as data:
        data = data[var_name]
#    # reverse the latitude dimension to make it increasing
#    data = data.reindex(lat=list(reversed(data.lat)))
#    # select the region of interest, this will generate a dataset with shape (1590, 64, 64)
#    data = data.sel(lon=slice(296, 360), lat=slice(0, 64))
    return data


def split_data(data, split='train'):
    """split data into training, validation and testing sets

    Args:
        data: a xr.dataarray of shape (1590, 64, 64)
        split: a string, either 'train', 'valid' or 'test'

    Returns:
        One of the following:
        train_data: a numpy array of shape (50*12, 64, 64) = (600, 64, 64) # 50 years
        valid_data: a numpy array of shape (10*12, 64, 64) = (120, 64, 64) # 10 years
        test_data: a numpy array of shape (5*12, 64, 64) = (60, 64, 64) # 5 years
    """
#    ds = xr.open_dataset('./saved_aod_20230101.nc')
#    ds_aod= ds.variables['aod']
#    train_data = data.values[:600,:,:]
#    valid_data = data.values[600:720,:,:]
#    test_data = data.values[720:780,:,:]

#    train_data = data.sel(time=slice("1950-01-01",
#                                        "1999-12-31")).values
#    valid_data = data.sel(time=slice("2000-01-01",
#                                        "2010-12-31")).values
#    test_data = data.sel(time=slice("2011-01-01",
#                                        "2015-12-31")).values
    if split == 'train':
        train_data = data.values[:9600,:,:]
        return train_data
    elif split == 'valid':
        valid_data = data.values[9600:12000,:,:]
        return valid_data
    elif split == 'test':
        test_data = data.values[12000:13200,:,:]
        return test_data
    else:
        raise ValueError('Invalid split name: {}'.format(split))


class NcDataset(Dataset):
    def __init__(self, config, split='train'):
        super().__init__()
        self.datas = load_nc_data(config)
        self.input_size = config.input_size
        self.num_frames_input = config.num_frames_input
        self.num_frames_output = config.num_frames_output
        self.num_frames = config.num_frames_input + config.num_frames_output
        self.datas = split_data(self.datas, split)
        print('Loaded {} samples ({})'.format(self.__len__(), split))

    def __getitem__(self, index):
#        data = self.datas[index:index+self.num_frames]
        data = self.datas[index*self.num_frames:(index+1)*self.num_frames]
        inputs = data[:self.num_frames_input]
        targets = data[self.num_frames_input:]
        inputs = inputs[..., np.newaxis]
        targets = targets[..., np.newaxis]
        # replace nan values with 0
        # inputs = np.nan_to_num(inputs)
        # targets = np.nan_to_num(targets)
        inputs = torch.from_numpy(inputs).permute(0, 3, 1, 2).float().contiguous()
        targets = torch.from_numpy(targets).permute(0, 3, 1, 2).float().contiguous()
        return inputs, targets
    

    def __len__(self):
        return self.datas.shape[0] // self.num_frames


if __name__ == "__main__":
    from configs.config_3x3_16_3x3_32_3x3_64_nc import config
    data = load_nc_data(config)
    print(data.shape)
    print(data.time.values[0])
    print(data.time.values[-1])
