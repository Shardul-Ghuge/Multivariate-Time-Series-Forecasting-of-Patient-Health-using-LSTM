import numpy as np
import os

model_type = "base_lstm"
#model_type = "seq2seq"
#model_type = "teacher_forcing"
#model_type = "attention"

MORTALITY_RATE = 0.4
upper_threshold = 0.8
lower_threshold = 0.1
BASE_PATH = "C:\\Users\\ghuge\\Desktop\\UofT\\Thesis\\Predicting-future-medical-diagnoses-with-LSTM\\model\\trained_models"
DATA_PATH = os.path.join(BASE_PATH, model_type,"mimic","with-mortality", "mortality-rate-{}".format(MORTALITY_RATE))

print("MORTALITY ACCURACY FOR THE CASE WHEN MORTALITY RATE IS:", MORTALITY_RATE)

#load the prediction and target data
predictions = np.load(os.path.join(DATA_PATH,'predictions.npy'))
targets = np.load(os.path.join(DATA_PATH,'targets.npy'))

targets_mortality = targets[:, :, 82]
predictions_mortality = predictions[:, :, 82]

print("Total number of patients: ", targets_mortality.shape[0])

#find how many entries in predictions_mortality are over theresold
print("predictions_mortality > {}: ".format(upper_threshold), np.sum(predictions_mortality > upper_threshold))
print("targets_mortality > {}: ".format(upper_threshold), np.sum(targets_mortality > lower_threshold))
print("predictions_mortality < {}: ".format(lower_threshold), np.sum(predictions_mortality < lower_threshold))
print("targets_mortality < {}: ".format(lower_threshold), np.sum(targets_mortality < lower_threshold))

#set all below lower_threshold in predictions_mortality to 0
predictions_mortality[predictions_mortality < lower_threshold] = 0

#set all above upper_threshold in predictions_mortality to 1
predictions_mortality[predictions_mortality > upper_threshold] = 1

# print("targets_mortality: ", targets_mortality[79])
# print("predictions_mortality: ", predictions_mortality[79])

# #Compare predicted and target mortality arrays element-wise
# correct = predictions_mortality == targets_mortality

# Compare predicted and target mortality arrays for only last time step
correct = predictions_mortality[:, -1] == targets_mortality[:, -1]
print("correct: ", correct.shape)
print("correct: ", correct)

# Calculate accuracy as the fraction of correct predictions
accuracy = np.mean(correct)

print(f"Accuracy: {accuracy:.4f}")
