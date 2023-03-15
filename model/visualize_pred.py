import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_predictions(inputs, predictions, targets, patient_num, RESULTS_PATH):
    # Plot the progression of features 1, 2 and 3 across all 49 time steps for the selected patient
    num_timesteps = 49
    feats = [0, 1, 2]
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle("Prediction Visualization for Patient {}".format(patient_num))

    #feats[0]
    axs[0].plot(range(1, num_timesteps), inputs[patient_num,:, feats[0]], label='Input', color='blue')
    axs[0].plot(range(1, num_timesteps), targets[patient_num,:, feats[0]], label='Target', linestyle='--', color='red')
    axs[0].plot(range(1, num_timesteps), predictions[patient_num,:, feats[0]], label='Prediction', linestyle='--', color='green')
    axs[0].set_xlabel('Time step')
    axs[0].set_ylabel('Feature {}'.format(feats[0]))
    axs[0].legend()

    #feats[1]
    axs[1].plot(range(1, num_timesteps), inputs[patient_num,:, feats[1]], label='Input', color='blue')
    axs[1].plot(range(1, num_timesteps), targets[patient_num,:, feats[1]], label='Target', linestyle='--', color='red')
    axs[1].plot(range(1, num_timesteps), predictions[patient_num,:, feats[1]], label='Prediction', linestyle='--', color='green')
    axs[1].set_xlabel('Time step')
    axs[1].set_ylabel('Feature {}'.format(feats[1]))
    axs[1].legend()

    #feats[2]
    axs[2].plot(range(1, num_timesteps), inputs[patient_num,:, feats[2]], label='Input', color='blue')
    axs[2].plot(range(1, num_timesteps), targets[patient_num,:, feats[2]], label='Target', linestyle='--', color='red')
    axs[2].plot(range(1, num_timesteps), predictions[patient_num,:, feats[2]], label='Prediction', linestyle='--', color='green')
    axs[2].set_xlabel('Time step')
    axs[2].set_ylabel('Feature {}'.format(feats[2]))
    #axs[2].set_ylabel('Mortality (0: Alive, 1: Dead)')
    axs[2].legend()

    plt.tight_layout()
    pic_path = os.path.join(RESULTS_PATH, "prediction_visualization_patient_{}.png".format(patient_num))
    plt.savefig(pic_path)

if __name__ == "__main__":
    model_type = "base_lstm"
    #model_type = "seq2seq"
    num_patients = 5000
    MODEL_PATH = "C:\\Users\\ghuge\\Desktop\\UofT\\Thesis\\Predicting-future-medical-diagnoses-with-LSTM\\model\\trained_models"
    SYNTHETIC_RESULTS_PATH = os.path.join(MODEL_PATH, model_type, "synthetic-data", "patients-{}".format(num_patients))

    #load the prediction and target data
    inputs = np.load(os.path.join(SYNTHETIC_RESULTS_PATH,'inputs.npy'))
    predictions = np.load(os.path.join(SYNTHETIC_RESULTS_PATH,'predictions.npy'))
    targets = np.load(os.path.join(SYNTHETIC_RESULTS_PATH,'targets.npy'))

    # Visualize the predictions for the first k patients
    count = 5
    for i in range(417,421):
        visualize_predictions(inputs, predictions, targets, i, SYNTHETIC_RESULTS_PATH)

