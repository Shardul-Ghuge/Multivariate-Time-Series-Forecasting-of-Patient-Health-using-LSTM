import torch
import os 
import pandas as pd

#load the mortality data as dataframe
PROCESSED_DATA_DIR = os.path.join(os.curdir,"processed_data")
train_mortality_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "LSTM_death_tags_train.csv"))[["unique_id","value"]] 
val_mortality_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "LSTM_death_tags_val.csv"))[["unique_id","value"]]
test_mortality_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "LSTM_death_tags_test.csv"))[["unique_id","value"]]

#load the LSTM train, val and test datasets
train_tensor = torch.load(os.path.join(os.curdir,"processed_tensors","LSTM_tensor_train.pt"))
val_tensor = torch.load(os.path.join(os.curdir,"processed_tensors","LSTM_tensor_val.pt"))
test_tensor = torch.load(os.path.join(os.curdir,"processed_tensors","LSTM_tensor_test.pt"))

#fill in the nan values with 0 for the train, val and test datasets
train_tensor = torch.nan_to_num(train_tensor).float()
val_tensor = torch.nan_to_num(val_tensor).float()
test_tensor = torch.nan_to_num(test_tensor).float()

#add an extra feature to the train, val and test datasets
train_tensor = torch.cat((train_tensor, torch.zeros(train_tensor.shape[0], train_tensor.shape[1], 1)), dim=2)
val_tensor = torch.cat((val_tensor, torch.zeros(val_tensor.shape[0], val_tensor.shape[1], 1)), dim=2)
test_tensor = torch.cat((test_tensor, torch.zeros(test_tensor.shape[0], test_tensor.shape[1], 1)), dim=2)

#convert the value column to a tensor
train_mortality_tensor = torch.tensor(train_mortality_df["value"].values).float()
val_mortality_tensor = torch.tensor(val_mortality_df["value"].values).float()
test_mortality_tensor = torch.tensor(test_mortality_df["value"].values).float()

# add the mortality tensor to the train, val and test datasets
train_tensor[:, -1, -1] = train_mortality_tensor
val_tensor[:, -1, -1] = val_mortality_tensor
test_tensor[:, -1, -1] = test_mortality_tensor

#want last two time steps to have same mortality value for proper feature lagging
train_tensor[:, -2, -1] = train_mortality_tensor
val_tensor[:, -2, -1] = val_mortality_tensor
test_tensor[:, -2, -1] = test_mortality_tensor

#save the train, val and test datasets
torch.save(train_tensor, os.path.join(os.curdir,"processed_tensors","LSTM_mortality_train.pt"))
torch.save(val_tensor, os.path.join(os.curdir,"processed_tensors","LSTM_mortality_val.pt"))
torch.save(test_tensor, os.path.join(os.curdir,"processed_tensors","LSTM_mortality_test.pt"))




