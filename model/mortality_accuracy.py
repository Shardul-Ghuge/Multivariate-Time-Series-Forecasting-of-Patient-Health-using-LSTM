import numpy as np

model_type = "base_lstm"
#model_type = "seq2seq"

#load the prediction and target data
predictions = np.load('trained_models/{}/predictions.npy'.format(model_type))
targets = np.load('trained_models/{}/targets.npy'.format(model_type))

targets_mortality = targets[:, :, 82]
predictions_mortality = predictions[:, :, 82]

#set all below 0.05 in predictions_mortality to 0
predictions_mortality[predictions_mortality < 0.05] = 0

#set all above 0.95 in predictions_mortality to 1
predictions_mortality[predictions_mortality > 0.95] = 1

# print("targets_mortality: ", targets_mortality[79])
# print("predictions_mortality: ", predictions_mortality[79])

#find how many entries in predictions_mortality are over theresold
print("predictions_mortality > 0.95: ", np.sum(predictions_mortality > 0.95))
print("targets_mortality > 0.95: ", np.sum(targets_mortality > 0.95))

# Compare predicted and target mortality arrays element-wise
correct = predictions_mortality == targets_mortality

# # Compare predicted and target mortality arrays for only last time step
# correct = predictions_mortality[:, -1] == targets_mortality[:, -1]

# Calculate accuracy as the fraction of correct predictions
accuracy = np.mean(correct)

print(f"Accuracy: {accuracy:.4f}")


