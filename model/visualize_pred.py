import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_predictions(inputs, predictions, targets, patient_num, RESULTS_PATH):
    # Plot the progression of features 1, 2 and 3 across all 49 time steps for the selected patient
    num_timesteps = 49
    feats = [0, 1, 2, 17, 82]
    fig, axs = plt.subplots(5, 1, figsize=(10, 13))
    fig.suptitle("Prediction Visualization for Patient {}".format(patient_num))

    # Loop through subplots
    for i in range(5):
        # feats[0]
        axs[i].plot(range(1, num_timesteps), targets[patient_num, :, feats[i]], label='Target', color='red')
        axs[i].plot(range(1, num_timesteps), predictions[patient_num, :, feats[i]], label='Prediction', linestyle='--', color='blue')
        axs[i].set_xlabel('Time step')
        #axs[i].set_ylabel('Feature {}'.format(feats[i]))
        if i == 4:
            axs[i].set_ylabel('Mortality (0: Alive, 1: Dead)')
        else:
            axs[i].set_ylabel('Feature {}'.format(feats[i]))
        axs[i].legend()

    plt.tight_layout()
    pic_path = os.path.join(RESULTS_PATH, "prediction_visualization_patient_{}.png".format(patient_num))
    plt.savefig(pic_path)

if __name__ == "__main__":
    model_type = "base_lstm"
    #model_type = "seq2seq"
    #model_type = "teacher_forcing"
    #model_type = "attention"

    num_patients = 39481
    MORTALITY_RATE = 0.4
    MODEL_PATH = "C:\\Users\\ghuge\\Desktop\\UofT\\Thesis\\Predicting-future-medical-diagnoses-with-LSTM\\model\\trained_models"
    SYNTHETIC_RESULTS_PATH = os.path.join(MODEL_PATH, model_type, "synthetic-data", "patients-{}".format(num_patients))
    MIMIC_RESULTS_PATH = os.path.join(MODEL_PATH, model_type,"mimic","with-mortality", "mortality-rate-{}".format(MORTALITY_RATE))
    
    #load the prediction and target data
    inputs = np.load(os.path.join(SYNTHETIC_RESULTS_PATH,'inputs.npy'))
    predictions = np.load(os.path.join(SYNTHETIC_RESULTS_PATH,'predictions.npy'))
    targets = np.load(os.path.join(SYNTHETIC_RESULTS_PATH,'targets.npy'))

    # Visualize the predictions for the first k patients
    count = 5
    for i in range(0,5):
        visualize_predictions(inputs, predictions, targets, i, SYNTHETIC_RESULTS_PATH)

