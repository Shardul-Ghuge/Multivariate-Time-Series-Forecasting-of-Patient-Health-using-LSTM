import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Input data parameters
num_patients = 5000
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
    X = data[:, 1:, :]
    y = data[:, :-1, :]
    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).float().to(device)

    #save the data
    print("X", X.shape, "y", y.shape)
    torch.save(X, "size-{}-X_correlation1_2_rest_0.pt".format(num_patients))
    torch.save(y, "size-{}-y_correlation1_2_rest_0.pt".format(num_patients))

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
    X = data[:, 1:, :]
    y = data[:, :-1, :]
    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).float().to(device)

    #save the data
    print("X", X.shape, "y", y.shape)
    torch.save(X, "size-{}-X_correlation1_2_rest_random.pt".format(num_patients))
    torch.save(y, "size-{}-y_correlation1_2_rest_random.pt".format(num_patients))

def clipped_correlation1_2_rest_random(num_patients, num_timesteps, num_features):
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
                #feature_k = 
                data[i, j, k] = np.random.uniform(0, 1)

    #X is lagged by 1 timestep relative to y
    X = data[:, 1:, :]
    y = data[:, :-1, :]
    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).float().to(device)

    #save the data
    print("X", X.shape, "y", y.shape)
    torch.save(X, "size-{}-X_correlation1_2_rest_random.pt".format(num_patients))
    torch.save(y, "size-{}-y_correlation1_2_rest_random.pt".format(num_patients))


if __name__ == "__main__":
    #correlation1_2_rest_0(num_patients, num_timesteps, num_features)
    correlation1_2_rest_random(num_patients, num_timesteps, num_features)