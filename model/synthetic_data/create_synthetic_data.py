import torch
import numpy as np
import os
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_DIR = "C:\\Users\\ghuge\\Desktop\\UofT\\Thesis\\Predicting-future-medical-diagnoses-with-LSTM\\model\\synthetic_data"

#Input data parameters
num_patients = 39481 #size of MIMIC-set that we have
num_timesteps = 49
num_features = 83

def correlation1_2_rest_0(num_patients, num_timesteps, num_features):
    #Generate a sysnthetic dataset for testing LSTM
    data = np.zeros((num_patients, num_timesteps, num_features))
    for i in range(num_patients):
        for j in range(num_timesteps - 1):
            #feature_1 will have a value between 0.5 and 1 with a probability of 30%, and a value of 0 with a probability of 70%
            if np.random.rand() < 0.1:
                feature_1 = np.random.uniform(0.5, 1)
            else:
                feature_1 = 0
            data[i, j, 0] = feature_1
            
            if feature_1 > 0.5:
                data[i, j+1, 1] = 1 #feature_2 will be 1 next timestep if feature_1 is greater than 0.5 now

    #X is lagged by 1 timestep relative to y
    X = data[:, :-1, :]
    y = data[:, 1:, :]
    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).float().to(device)

    #save the data
    print("X", X.shape, "y", y.shape)
    x_path = os.path.join(DATA_DIR, "size-{}-X_correlation1_2_rest_0.pt".format(num_patients))
    y_path = os.path.join(DATA_DIR, "size-{}-y_correlation1_2_rest_0.pt".format(num_patients))
    torch.save(X, x_path)
    torch.save(y, y_path)

def correlation1_2_rest_random(num_patients, num_timesteps, num_features):
    #Generate a sysnthetic dataset for testing LSTM
    data = np.zeros((num_patients, num_timesteps, num_features))
    for i in range(num_patients):
        for j in range(num_timesteps - 1):
            #feature_1 will have a value between 0.5 and 1 with a probability of 30%, and a value of 0 with a probability of 70%
            if np.random.rand() < 0.1:
                feature_1 = np.random.uniform(0.5, 1)
            else:
                feature_1 = 0
            data[i, j, 0] = feature_1
            
            if feature_1 > 0.5:
                data[i, j+1, 1] = 1 #feature_2 will be 1 next timestep if feature_1 is greater than 0.5 now
            
            # features 3 onwards will have a random value between 0 and 1
            for k in range(2, num_features):
                data[i, j, k] = np.random.uniform(0, 1)

    #X is lagged by 1 timestep relative to y
    X = data[:, :-1, :]
    y = data[:, 1:, :]
    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).float().to(device)

    #save the data
    print("X", X.shape, "y", y.shape)
    x_path = os.path.join(DATA_DIR, "size-{}-X_correlation1_2_rest_random.pt".format(num_patients))
    y_path = os.path.join(DATA_DIR, "size-{}-y_correlation1_2_rest_random.pt".format(num_patients))
    torch.save(X, x_path)
    torch.save(y, y_path)


def correlation1_2_some_random(num_patients, num_timesteps, num_features, num_random_features = 10):
    #Generate a sysnthetic dataset for testing LSTM
    data = np.zeros((num_patients, num_timesteps, num_features))
    for i in range(num_patients):
        for j in range(num_timesteps - 1):
            #feature_1 will have a value between 0.5 and 1 with a probability of 30%, and a value of 0 with a probability of 70%
            if np.random.rand() < 0.1:
                feature_1 = np.random.uniform(0.5, 1)
            else:
                feature_1 = 0
            data[i, j, 0] = feature_1
            
            if feature_1 > 0.5:
                data[i, j+1, 1] = 1 #feature_2 will be 1 next timestep if feature_1 is greater than 0.5 now
            
            # Generate exactly 10 random feature indices between feat3 (2) and feat83 (82)
            
            random_feature_indices = random.sample(range(2, num_features), num_random_features)

            for k in range(2, num_features):
                if k in random_feature_indices:
                    data[i, j, k] = np.random.uniform(0, 1)
                else:
                    data[i, j, k] = 0

    #X is lagged by 1 timestep relative to y
    X = data[:, :-1, :]
    y = data[:, 1:, :]
    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).float().to(device)

    #save the data
    print("X", X.shape, "y", y.shape)
    x_path = os.path.join(DATA_DIR, "size-{}-X_correlation1_2_{}_random.pt".format(num_patients, num_random_features))
    y_path = os.path.join(DATA_DIR, "size-{}-y_correlation1_2_{}_random.pt".format(num_patients, num_random_features))
    torch.save(X, x_path)
    torch.save(y, y_path)

if __name__ == "__main__":
    #correlation1_2_rest_0(num_patients, num_timesteps, num_features)
    #correlation1_2_rest_random(num_patients, num_timesteps, num_features)
    correlation1_2_some_random(num_patients, num_timesteps, num_features, num_random_features = 68)