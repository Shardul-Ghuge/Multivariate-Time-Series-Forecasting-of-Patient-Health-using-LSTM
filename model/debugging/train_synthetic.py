import torch
import torch.nn as nn
import numpy as np
from MultiOutputLSTM import MultiOutputLSTM
import matplotlib.pyplot as plt

# check if GPU is available
if(torch.cuda.is_available()):
    print('Training on GPU!')
else: 
    print('Training on CPU!')
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Generate a sysnthetic dataset for testing LSTM
num_patients = 50
num_timesteps = 49
num_features = 83

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

model = MultiOutputLSTM(input_size=num_features, hidden_size=256, num_layers=3, output_size=num_features).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    outputs = model(X)
    loss = criterion(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Test the model and visualize predictions
patient_idx = 34 # Select a patient to visualize

# Generate predictions for the selected patient
with torch.no_grad():
    model.eval()
    inputs = X[patient_idx].unsqueeze(0)
    outputs = model(inputs)

# Convert tensors to numpy arrays
inputs = inputs.cpu().detach().numpy()[0]
outputs = outputs.cpu().detach().numpy()[0]
targets = y.cpu().detach().numpy()[patient_idx]

# Plot the progression of features 1, 2 and 3 across all 49 time steps
fig, axs = plt.subplots(3, 1, figsize=(10, 8))
fig.suptitle('Multi-Output LSTM Prediction for Patient {}'.format(patient_idx))
#feature 1
axs[0].plot(range(1, num_timesteps), inputs[:, 0], label='Input', color='blue')
axs[0].plot(range(1, num_timesteps), targets[:, 0], label='Target', linestyle='--', color='red')
axs[0].plot(range(1, num_timesteps), outputs[:, 0], label='Prediction', linestyle='--', color='green')
axs[0].set_xlabel('Time step')
axs[0].set_ylabel('Feature 1')
axs[0].legend()

#feature 2
axs[1].plot(range(1, num_timesteps), inputs[:, 1], label='Input', color='blue')
axs[1].plot(range(1, num_timesteps), targets[:, 1], label='Target', linestyle='--', color='red')
axs[1].plot(range(1, num_timesteps), outputs[:, 1], label='Prediction', linestyle='--', color='green')
axs[1].set_xlabel('Time step')
axs[1].set_ylabel('Feature 2')
axs[1].legend()

#feature 2
axs[2].plot(range(1, num_timesteps), inputs[:, 47], label='Input', color='blue')
axs[2].plot(range(1, num_timesteps), targets[:, 47], label='Target', linestyle='--', color='red')
axs[2].plot(range(1, num_timesteps), outputs[:, 47], label='Prediction', linestyle='--', color='green')
axs[2].set_xlabel('Time step')
axs[2].set_ylabel('Feature 47')
axs[2].legend()

plt.tight_layout()
plt.savefig('size-{}-prediction-random-vals-other-feats.png'.format(num_patients))
plt.show()
