import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import os
from LSTM import *
from LaggedDataset import *
from datetime import datetime
import matplotlib.pyplot as plt

#DATA_DIR = "~/Desktop/UofT/Thesis/Predicting-future-medical-diagnoses-with-LSTM/preprocessing/processed_tensors/"
DATA_DIR = "C:\\Users\\ghuge\\Desktop\\UofT\\Thesis\\Predicting-future-medical-diagnoses-with-LSTM\\preprocessing\\processed_tensors"
MODEL_PATH = "C:\\Users\\ghuge\\Desktop\\UofT\\Thesis\\Predicting-future-medical-diagnoses-with-LSTM\\model\\trained_models"
#model_type = "base_lstm"
model_type = "seq2seq"

def get_model(model, model_params):
    models = {
        "base_lstm": LSTM,
        "seq2seq": Seq2Seq,
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

#load the LSTM train, val and test datasets
train_tensor = torch.load(os.path.join(DATA_DIR,"LSTM_tensor_train.pt"))
val_tensor = torch.load(os.path.join(DATA_DIR,"LSTM_tensor_val.pt"))
test_tensor = torch.load(os.path.join(DATA_DIR,"LSTM_tensor_test.pt"))

#fill in the nan values with 0 for the train, val and test datasets
train_tensor = torch.nan_to_num(train_tensor).float()
val_tensor = torch.nan_to_num(val_tensor).float()
test_tensor = torch.nan_to_num(test_tensor).float()

print("train_tensor", train_tensor.shape)

# creating the train, val and test loaders
lags = 5 # Define the number of lags
batch_size = 64 # Define the batch size

#calls the lagged dataset class which sets up the lagged data for the LSTM model with lag = `lags`
train_loader = get_data_loaders(train_tensor, lags, batch_size)
val_loader = get_data_loaders(val_tensor, lags, batch_size)
test_loader = get_data_loaders(test_tensor, lags, batch_size)

print("Done loading the data!")

#set up the training, validation and testing procedure
class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []
    
    def train_step(self, x, y):
        #define the mask so that the gradients of the zero elements will be set to zero and will not be used to update the model's parameters
        mask = (x != 0).float()

        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss and applies mask to zero out loss for padded elements
        loss = self.loss_fn(y, yhat) * mask

        loss = torch.mean(loss) #When reduction='none', the MSELoss function will return a tensor of the same shape as the input, with the loss value computed for each element. Thats why we need to take the mean of the loss tensor

        # Computes gradients
        loss.backward()

        # # Apply mask to gradients: not neeed if masked loss is setup for MSE
        # for param in self.model.parameters():
        #     print("param.grad", param.grad.shape, "mask", mask.shape)
        #     param.grad *= mask

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, batch_size=256, n_epochs=50, n_features=82):
        model_path = os.path.join(MODEL_PATH, model_type, f'{model_type}_best_model_{datetime.now().strftime("%Y-%m-%d")}.pth')   
        best_val_loss = float('inf')
        
        print("started training")
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    mask = (x_val != 0).float()
                    val_loss = self.loss_fn(y_val, yhat)* mask
                    val_loss = torch.mean(val_loss).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

            if (epoch <= 10) | (epoch % 10 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.6f}\t Validation loss: {validation_loss:.6f}"
                )

            #Only save the model if the validation loss is the lowest
            if validation_loss < best_val_loss:
                best_val_loss = validation_loss
                best_epoch = epoch
                torch.save(self.model.state_dict(), model_path)
        
        print("Best {} model saved at lowest validation set error of {} at epoch {}".format(model_type ,np.round(best_val_loss,6), best_epoch)) 
        
    def evaluate(self, test_loader, batch_size, n_features):
        '''
        To visualize the predicted and target values for the multi-output case across all time steps, 
        you would need to concatenate the predicted and target values from each batch into a single 3D array. 
        Thats why we create a prediction_batch and target_batch list. Then, you can concatenate the predicted and target 
        values from all batches into a single 3D numpy array using the np.concatenate function
        '''
        
        print("started evaluation")
        with torch.no_grad():
            batch_test_losses = []
            count = 0
            prediction_batch = []
            target_batch = []
            for x_test, y_test in test_loader:
                count += 1
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                # print("y_test", y_test)
                # print("y_hat_test", yhat)
                # print("y_test", y_test.shape)
                # print("y_hat_test", yhat.shape)
                predicted  = yhat.cpu().detach().numpy()
                target = y_test.cpu().detach().numpy()
                prediction_batch.append(predicted)
                target_batch.append(target)   
                mask = (x_test != 0).float()                    
                MSE_loss = self.loss_fn(y_test, yhat) * mask
                MSE_loss = torch.mean(MSE_loss).item()
                batch_test_losses.append(MSE_loss)
            
            test_loss = np.mean(batch_test_losses)

        print(f"Mean Test loss over {count} batches: {test_loss:.4f}")
        
        prediction_batch = np.concatenate(prediction_batch, axis=0)
        target_batch = np.concatenate(target_batch, axis=0)
        
        return prediction_batch, target_batch
    
    def plot_losses(self):
        plt.figure(1)
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses for {} model".format(model_type))
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig('trained_models/{}/train_and_val_loss.png'.format(model_type))


#define the LSTM/Seq2Seq model parameters
input_size = 82 #number of features
hidden_size = 64 #related to latent_variables of the model
num_layers = 3 #depth of the model
output_size = 82 #same as input_size becasue we are predicting all 82 variables
dropout_rate = 0.2

n_epochs = 30
learning_rate = 1e-3
weight_decay = 1e-6

model_params = {'input_size': input_size,
                'hidden_size' : hidden_size,
                'num_layers' : num_layers,
                'output_size' : output_size,
                'dropout_rate' : dropout_rate}

# Define your model, loss and optimizer
model = get_model(model_type, model_params).to(device)
print(model)
loss_fn  = torch.nn.MSELoss(reduction='none')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=n_epochs, n_features=input_size)
opt.plot_losses()
predictions, targets = opt.evaluate(test_loader, batch_size=batch_size, n_features=input_size)

#save the predictions and targets as numpy arrays for future use
np.save('trained_models/{}/predictions.npy'.format(model_type), predictions)
np.save('trained_models/{}/targets.npy'.format(model_type), targets)

print("Done training and evaluating the model!")

def visualize_predictions(predictions, targets):
    #select the sample and feature from the array
    sample_num = 512
    feature_num = 27

    # Plot the predicted values and target values for the selected sample and feature
    plt.figure(2)
    plt.plot(predictions[sample_num, :, feature_num], label='Predicted')
    plt.plot(targets[sample_num, :, feature_num], label='Target')
    plt.xlabel('Time step')
    plt.ylabel('Value')
    plt.title('{} Predicted and target values for sample {} and feature {}'.format(model_type ,sample_num, feature_num))
    plt.legend()
    plt.savefig('trained_models/{}/predictions_vs_trueValues_sample_{}_feature_{}.png'.format(model_type ,sample_num, feature_num))
    print("Done plotting the predictions and targets!")

def calculate_performance_metrics(predictions, targets):
    '''
    1. Mean Squared Error (MSE) measures the average squared difference between the predicted and actual values. Lower values indicate a better fit. 

    2. Mean Absolute Error (MAE) is a measure of the average magnitude of the errors in a set of predictions, without considering their direction. 
    It is calculated as the average absolute difference between predicted and actual values.

    3. Mean Absolute Percentage Error (MAPE) is a measure of the average percentage error of predictions. 
    It is calculated as the average absolute percentage difference between predicted and actual values.

    4. Root Mean Squared Error (RMSE) is a measure of the average magnitude of the errors in a set of predictions, with a higher weight for larger errors.
    It is calculated as the square root of the average of the squared differences between predicted and actual values.

    5. Coefficient of Determination (R²) is a measure of how well the predictions fit the actual values. It is a value between 0 and 1, where 1 represents a perfect fit.

    6. Correlation coefficient is a measure of the linear relationship between two variables. 
    It ranges from -1 to 1, where 1 represents a perfect positive correlation, 0 represents no correlation, and -1 represents a perfect negative correlation.
    '''

    #calculate mean squared error (MSE)
    mse = ((predictions - targets) ** 2).mean()
    print("MSE:", mse)

    #calculate mean absolute error (MAE)
    mae = np.abs(predictions - targets).mean()
    print("MAE:", mae)

    # # calculate mean absolute percentage error (MAPE)
    # mape = (np.abs((targets - predictions) / targets) * 100).mean()
    # print("MAPE:", mape)

    # calculate root mean squared error (RMSE)
    rmse = np.sqrt(mse)
    print("RMSE:", rmse)

    # calculate coefficient of determination (R^2): 
    r2 = 1 - ((targets - predictions) ** 2).sum() / ((targets - targets.mean()) ** 2).sum()
    print("R^2:", r2)


visualize_predictions(predictions, targets)
calculate_performance_metrics(predictions, targets)

