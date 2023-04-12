import torch
import os
DATA_DIR = "C:\\Users\\ghuge\\Desktop\\UofT\\Thesis\\Predicting-future-medical-diagnoses-with-LSTM\\preprocessing\\processed_tensors"

#load the LSTM train, val and test datasets
train_tensor = torch.load(os.path.join(DATA_DIR,"LSTM_mortality_train_last.pt"))
val_tensor = torch.load(os.path.join(DATA_DIR,"LSTM_mortality_val_last.pt"))
test_tensor = torch.load(os.path.join(DATA_DIR,"LSTM_mortality_test_last.pt"))

all_tensors = [train_tensor, val_tensor, test_tensor] #need to duplicate mortality seperately to prevent data leakage
type = ["train", "val", "test"]

# all_tensors = [train_tensor]
# type = ["train"]
count = 0

for data in all_tensors:
    print("\nDataset shape for {} set:".format(type[count]), data.shape)
    # count number of Mortality = 1 patients in data
    mortality_1_count = torch.sum(data[:, -1, -1] == 1)
    print("mortality count before duplication: ", mortality_1_count)

    # calculate number of additional samples needed to achieve desired mortality ratio
    total_count = data.shape[0]
    desired_mortality_ratio = 0.65
    desired_mortality_count = int(total_count * desired_mortality_ratio)
    additional_mortality_count = desired_mortality_count - mortality_1_count
    print("additional mortality count: ", additional_mortality_count)

    # create subset of train dataset that only contains Mortality = 1 patients
    mortality_1_subset = data[data[:, -1, -1] == 1]
    print("mortality_1_subset shape: ", mortality_1_subset.shape)
    #print("mortality_1_subset: ", mortality_1_subset)

    # repeat Mortality = 1 subset enough times to achieve desired mortality count on axis 0 only
    repeated_mortality_1_subset = mortality_1_subset.repeat(additional_mortality_count // mortality_1_count + 1, 1, 1)

    # concatenate original train tensor with repeated Mortality = 1 subset
    data = torch.cat((data, repeated_mortality_1_subset[:additional_mortality_count]), dim=0)

    #remove additional_mortality_count patients with mortality = 0 to preserve the same patient count
    data = data[torch.randperm(data.shape[0])]
    data = data[additional_mortality_count:]

    #check the number of patients who have a mortality label of 1
    print("mortality count after duplication", torch.sum(data[:, -1, -1]).item())

    #print number of patients in each dataset
    print("Dataset size", data.shape[0])

    #print Mortality ratio rounded to 2 decimal places
    final_mortality_ratio = round(torch.sum(data[:, -1, -1]).item() / data.shape[0], 2)
    print("Mortality ratio: ", final_mortality_ratio)

    print("Dataset shape after duplication: ", data.shape)

    #save the datasets to the processed_tensors folder
    torch.save(data, os.path.join(DATA_DIR, "{}-mortality_ratio_{}.pt".format(final_mortality_ratio, type[count])))
    count += 1
    

