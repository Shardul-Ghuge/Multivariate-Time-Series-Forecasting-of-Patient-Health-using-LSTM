import torch
import os
DATA_DIR = "C:\\Users\\ghuge\\Desktop\\UofT\\Thesis\\Predicting-future-medical-diagnoses-with-LSTM\\preprocessing\\processed_tensors"

#load the LSTM train, val and test datasets
train_tensor = torch.load(os.path.join(DATA_DIR,"LSTM_mortality_train.pt"))
val_tensor = torch.load(os.path.join(DATA_DIR,"LSTM_mortality_val.pt"))
test_tensor = torch.load(os.path.join(DATA_DIR,"LSTM_mortality_test.pt"))

#combine train, val and test datasets
data = torch.cat((train_tensor, val_tensor, test_tensor), dim=0)
print("data-size: ", data.shape)

# count number of Mortality = 1 patients in data
mortality_1_count = torch.sum(data[:, -1, -1] == 1)
print("mortality count before duplication: ", mortality_1_count)

# calculate number of additional samples needed to achieve desired mortality ratio
total_count = data.shape[0]
desired_mortality_ratio = 0.4
desired_mortality_count = int(total_count * desired_mortality_ratio)
additional_mortality_count = desired_mortality_count - mortality_1_count

# create subset of train dataset that only contains Mortality = 1 patients
mortality_1_subset = data[data[:, -1, -1] == 1]

# repeat Mortality = 1 subset enough times to achieve desired mortality count on axis 0 only
repeated_mortality_1_subset = mortality_1_subset.repeat(additional_mortality_count // mortality_1_count + 1, 1, 1)

# concatenate original train tensor with repeated Mortality = 1 subset
data = torch.cat((data, repeated_mortality_1_subset[:additional_mortality_count]), dim=0)

#remove additional_mortality_count patients with mortality = 0 to preserve the same patient count
data = data[torch.randperm(data.shape[0])]
data = data[additional_mortality_count:]

#check the number of patients who have a mortality label of 1
print("mortality count", torch.sum(data[:, -1, -1]).item())

#print number of patients in each dataset
print("Dataset size", data.shape[0])

#print Mortality ratio rounded to 2 decimal places
final_mortality_ratio = round(torch.sum(data[:, -1, -1]).item() / data.shape[0], 2)
print("Mortality ratio: ", final_mortality_ratio)

#split the dataset into train, val and test (80%, 10%, 10%)
train_size = int(0.8 * data.shape[0])
val_size = int(0.1 * data.shape[0])
test_size = data.shape[0] - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.split(data, [train_size, val_size, test_size])

print("train size: ", train_dataset.shape)
print("val size: ", val_dataset.shape)
print("test size: ", test_dataset.shape)

#save the datasets to the processed_tensors folder
torch.save(train_dataset, os.path.join(DATA_DIR, "{}-mortality_ratio_train.pt".format(final_mortality_ratio)))
torch.save(val_dataset, os.path.join(DATA_DIR, "{}-mortality_ratio_val.pt".format(final_mortality_ratio)))
torch.save(test_dataset, os.path.join(DATA_DIR, "{}-mortality_ratio_test.pt".format(final_mortality_ratio)))

