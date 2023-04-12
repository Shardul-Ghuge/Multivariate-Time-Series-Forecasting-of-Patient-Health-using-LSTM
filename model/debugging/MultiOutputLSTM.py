#USEFUL to test Multivariate LSTM model before large scale predictions with MIMIC
import torch
import torch.nn as nn

# check if GPU is available
if(torch.cuda.is_available()):
    print('Training on GPU!')
else: 
    print('Training on CPU!')
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
This model takes as input a tensor of shape (batch_size, sequence_length, input_size), 
where batch_size is the number of patients, sequence_length is the number of timesteps (in this case, 49), 
and input_size is the number of features (in this case, 82). The model uses an LSTM layer with hidden_size units 
and num_layers layers, followed by a fully connected layer that outputs a tensor of shape 
(batch_size, sequence_length, output_size), where output_size is also 82 (since we are predicting the values of all features).
"""
class MultiOutputLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MultiOutputLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)       
        return out