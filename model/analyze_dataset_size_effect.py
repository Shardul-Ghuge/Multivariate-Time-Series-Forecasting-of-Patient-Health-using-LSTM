import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import os
from LSTM import *
from visualize_pred import *
from datetime import datetime
import matplotlib.pyplot as plt

DATA_DIR = "C:\\Users\\ghuge\\Desktop\\UofT\\Thesis\\Predicting-future-medical-diagnoses-with-LSTM\\preprocessing\\processed_tensors"
BASE_PATH = "C:\\Users\\ghuge\\Desktop\\UofT\\Thesis\\Predicting-future-medical-diagnoses-with-LSTM\\model\\dataset_size_effect"

model_type = "base_lstm"
#model_type = "seq2seq"
#model_type = "teacher_forcing"
#model_type = "attention"

def get_model(model, model_params):
    models = {
        "base_lstm": BASE_LSTM,
        "seq2seq": Seq2Seq,
        "teacher_forcing": Seq2Seq_TeacherForcing,
        "attention": Seq2Seq_Attention
    }
    return models.get(model.lower())(**model_params)

print("Working on {} model".format(model_type))

### set a random seed for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# check if GPU is available
if(torch.cuda.is_available()):
    print('Training on GPU!')
else: 
    print('Training on CPU!')
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def load_data(DATA_DIR, ratio):
    train_tensor = torch.load(os.path.join(DATA_DIR,"0.36-mortality_ratio_train.pt")) #replace filename to load other datasets from processed_tensors folder
    val_tensor = torch.load(os.path.join(DATA_DIR,"0.36-mortality_ratio_val.pt"))
    test_tensor = torch.load(os.path.join(DATA_DIR,"0.36-mortality_ratio_test.pt"))

    train_size = int(train_tensor.shape[0] * ratio)
    val_size = int(val_tensor.shape[0] * ratio)
    test_size = int(test_tensor.shape[0] * ratio)

    train_tensor = train_tensor[:train_size, :, :]
    val_tensor = val_tensor[:val_size, :, :]
    test_tensor = test_tensor[:test_size, :, :]

    return train_tensor, val_tensor, test_tensor


def analyze_dataset_size(min_ratio, max_ratio, step, n_epochs):
    ratios = np.arange(min_ratio, max_ratio, step)
    print("ratios: ", ratios)
    performance_metrics = []
    for ratio in ratios:
        #if "patients-{}" folder does not exist, create it
        if not os.path.exists(os.path.join(BASE_PATH, "dataset-ratio-{}".format(ratio))):
            os.makedirs(os.path.join(BASE_PATH, "dataset-ratio-{}".format(ratio))) 
        RESULTS_PATH = os.path.join(BASE_PATH, "dataset-ratio-{}".format(ratio))

        print("\n#############################################")
        print(f"\nAnalyzing dataset size with ratio {ratio}")

       #load the LSTM train, val and test datasets
        train_tensor, val_tensor, test_tensor = load_data(DATA_DIR, ratio)
        
        print("train tensor shape: ", train_tensor.shape)
        print("val tensor shape: ", val_tensor.shape)
        print("test tensor shape: ", test_tensor.shape)

        #split the train_tensor time series into x_train and y_train
        x_train = train_tensor[:, :-1, :]
        y_train = train_tensor[:, 1:, :]

        #split the val_tensor time series into x_val and y_val
        x_val = val_tensor[:, :-1, :]
        y_val = val_tensor[:, 1:, :]

        #split the test_tensor time series into x_test and y_test
        x_test = test_tensor[:, :-1, :]
        y_test = test_tensor[:, 1:, :]

        batch_size = 64

        #create train, val and test loaders
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=False, drop_last=True)
        val_loader   = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader  = DataLoader(TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False, drop_last=True)

        print("Done loading the data!")

        #define the LSTM/Seq2Seq model parameters
        input_size = 83 #number of features
        hidden_size = 256 #related to latent_variables of the model
        num_layers = 3 #depth of the model
        output_size = 83 #same as input_size becasue we are predicting all 82 variables
        dropout_rate = 0.2
        learning_rate = 1e-3
        weight_decay = 1e-6

        model_params = {'input_size': input_size,
                        'hidden_size' : hidden_size,
                        'num_layers' : num_layers,
                        'output_size' : output_size,
                        'dropout_rate' : dropout_rate}

        # Define your model, loss and optimizer
        model = get_model(model_type, model_params).to(device)
        loss_fn  = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer, RESULTS_PATH=RESULTS_PATH)
        opt.train(train_loader, val_loader, n_epochs=n_epochs)
        opt.plot_losses()
        inputs, predictions, targets = opt.evaluate(test_loader)

        #save the predictions and targets as numpy arrays for future use
        np.save(os.path.join(RESULTS_PATH, "inputs.npy"), inputs)
        np.save(os.path.join(RESULTS_PATH, "predictions.npy"), predictions)
        np.save(os.path.join(RESULTS_PATH, "targets.npy"), targets)

        print("Done training and evaluating the model!")

        # Calculate performance metrics for this dataset size
        mse = ((predictions - targets) ** 2).mean()
        mae = np.abs(predictions - targets).mean()
        rmse = np.sqrt(mse)
        r2 = 1 - ((targets - predictions) ** 2).sum() / ((targets - targets.mean()) ** 2).sum()

        performance_metrics.append({
            'ratio': ratio,
            'mse': mse,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        })

        for i in range(0,7):
            visualize_predictions(inputs, predictions, targets, i, RESULTS_PATH)

        print("Done visualizing the predictions!")
    
    return performance_metrics


#set up the training, validation and testing procedure
class Optimization:
    def __init__(self, model, loss_fn, optimizer, RESULTS_PATH):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.RESULTS_PATH = RESULTS_PATH
        self.train_losses = []
        self.val_losses = []
        
    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()

        if model_type == "teacher_forcing" or model_type == "attention":
            # Makes predictions
            yhat = self.model(x, y) #need y for teacher-forcing
        else:
            # Makes predictions
            yhat = self.model(x) #for BASE_LSTM

        # Computes loss
        loss = self.loss_fn(y, yhat)

        self.optimizer.zero_grad()

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        
        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, n_epochs):
        '''Has early stopping enabled so that when there more than 5 epochs 
        with no improvement in validation loss, the training stops.
        This helps prevent overfitting and saves time.
        '''
        model_path = os.path.join(self.RESULTS_PATH, f'{model_type}_best_model_{datetime.now().strftime("%Y-%m-%d")}.pth')   
        best_val_loss = float('inf')
        patience = 5
        epochs_without_improvement = 0
    
        print("started training")
        for epoch in range(1, n_epochs + 1):
            if epochs_without_improvement > patience:
                print(f"Early stopping at epoch {epoch}.")
                break

            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    if model_type == "teacher_forcing" or model_type == "attention":
                        yhat = self.model(x_val, y_val, teacher_forcing_ratio=0)
                    else:
                        yhat = self.model(x_val)

                    val_loss = self.loss_fn(y_val, yhat)
                    batch_val_losses.append(val_loss.item())
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 10 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.6f}\t Validation loss: {validation_loss:.6f}"
                )

            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                torch.save(self.model.state_dict(), model_path)
            else:
                epochs_without_improvement += 1

        print("Best {} model saved at lowest validation set error of {} at epoch {}".format(model_type, np.round(best_val_loss, 6), best_epoch)) 
        
    def evaluate(self, test_loader):
        '''
        To visualize the predicted and target values for the multi-output case across all time steps, 
        you would need to concatenate the predicted and target values from each batch into a single 3D array. 
        Thats why we create a prediction_batch and target_batch list. Then, you can concatenate the predicted and target 
        values from all batches into a single 3D numpy array using the np.concatenate function
        '''
        # Load the best saved model from training
        model_path = os.path.join(self.RESULTS_PATH, f'{model_type}_best_model_{datetime.now().strftime("%Y-%m-%d")}.pth')
        self.model.load_state_dict(torch.load(model_path))
        
        # Print the name of the best saved model
        print(f"Loaded the best saved model: {os.path.basename(model_path)}")

        print("started evaluation")
        with torch.no_grad():
            batch_test_losses = []
            input_batch = []
            prediction_batch = []
            target_batch = []
            count = 0
            for x_test, y_test in test_loader:
                count += 1
                x_test = x_test.to(device)
                y_test = y_test.to(device)
                self.model.eval()
                
                if model_type == "teacher_forcing" or model_type == "attention": 
                    # Makes predictions
                    yhat = self.model(x_test, y_test, teacher_forcing_ratio=0) #to make the model rely only on its own predictions
                else:
                    # Makes predictions
                    yhat = self.model(x_test) #for BASE_LSTM
                
                inputs = x_test.cpu().detach().numpy()
                predicted  = yhat.cpu().detach().numpy()
                target = y_test.cpu().detach().numpy()
                input_batch.append(inputs)
                prediction_batch.append(predicted)
                target_batch.append(target)                      
                MSE_loss = self.loss_fn(y_test, yhat)
                batch_test_losses.append(MSE_loss.item())
            
            test_loss = np.mean(batch_test_losses)

        print(f"Mean Test loss over {count} batches: {test_loss:.4f}")
        
        input_batch = np.concatenate(input_batch, axis=0)
        prediction_batch = np.concatenate(prediction_batch, axis=0)
        target_batch = np.concatenate(target_batch, axis=0)
        
        return input_batch, prediction_batch, target_batch
    
    def plot_losses(self):
        plt.figure()
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses for {} model".format(model_type))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        pic_path = os.path.join(self.RESULTS_PATH, 'train_and_val_loss.png') 
        plt.savefig(pic_path)

if __name__ == "__main__":
    min_ratio = 0.1
    max_ratio = 1.1
    step = 0.1
    n_epochs = 200

    performance_metrics  = analyze_dataset_size(min_ratio, max_ratio, step, n_epochs)

    performance_df = pd.DataFrame(performance_metrics)

    #save the performance metrics as a csv file
    performance_df.to_csv(os.path.join(BASE_PATH, "performance_metrics.csv"), index=False)

    plt.figure()
    plt.plot(performance_df['ratio'], performance_df['mse'], label='MSE')
    plt.plot(performance_df['ratio'], performance_df['mae'], label='MAE')
    plt.plot(performance_df['ratio'], performance_df['rmse'], label='RMSE')
    plt.plot(performance_df['ratio'], performance_df['r2'], label='R^2')
    plt.xlabel('Dataset Size Ratio')
    plt.ylabel('Performance Metrics')
    plt.legend()
    plt.title('Effect of Dataset Size on LSTM Model Performance')
    plt.show()
    pic_path = os.path.join(BASE_PATH, 'performance_plot.png') 
    plt.savefig(pic_path)