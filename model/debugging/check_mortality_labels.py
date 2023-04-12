import os
import torch

DATA_DIR = "C:\\Users\\ghuge\\Desktop\\UofT\\Thesis\\Predicting-future-medical-diagnoses-with-LSTM\\preprocessing\\processed_tensors"
train_tensor = torch.load(os.path.join(DATA_DIR,"0.4-mortality_ratio_train.pt"))
train_tensor = torch.load(os.path.join(DATA_DIR,"0.36-mortality_ratio_train.pt"))
print(train_tensor[:5, :, -1].shape)
print(train_tensor[5:15, :, -1])