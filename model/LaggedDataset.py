from torch.utils.data import Dataset, DataLoader
import torch

#NOTE: Experimental code, not used in the final model
class LaggedDataSet(Dataset):
    '''
    X is a tensor (num_samples - lags, lags, num_features) that represents a sequence of lags time steps for each sample, 
    and y is a tensor (num_samples - lags, num_features) that represents the target value for each sample, which is the 
    value at the next time step after the lags time steps. The purpose of this data 
    preparation is to train the seq2seq model to predict the next value for each feature, 
    given a sequence of previous values for the same feature.
    '''
    def __init__(self, data, lags):
        self.data = data
        self.lags = lags

        X = torch.zeros((data.shape[0] - lags, lags, data.shape[2]))
        y = torch.zeros((data.shape[0] - lags, data.shape[2]))
        for i in range(lags, data.shape[0]): #TODO: fix the shape issue
            X[i-lags, :, :] = data[i-lags:i, :, :].squeeze()
            y[i-lags, :] = data[i, :, :].squeeze()
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_data_loaders(data, lags, batch_size):
    dataset = LaggedDataSet(data, lags)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)