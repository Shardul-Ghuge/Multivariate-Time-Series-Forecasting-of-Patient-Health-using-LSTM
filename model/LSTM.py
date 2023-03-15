import torch

# check if GPU is available
if(torch.cuda.is_available()):
    print('Training on GPU!')
else: 
    print('Training on CPU!')
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#TODO: set up padding for the LSTM model so that it ignores zero values
#create an LSTM model 
class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout = dropout_rate)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Results in tensor with shape (batch_size, seq_length, output_size) representing the predicted time series for all time steps.
        out = self.fc(out)
        return out


class Seq2Seq(torch.nn.Module):
    '''
    The key change here is the availabilty of the masking function to handle the sparsity in our data. 
    The self.mask(x) line in forward applies a masking operation on the input tensor x. The mask function creates 
    a masking layer that masks all elements in the input tensor that have a value of 0. This means that when the model processes 
    the input tensor, it will treat all elements with a value of 0 as missing values, and will not take them into account when computing the output. 
    This is useful for handling sparse input tensors like the one we have, where there are many time steps where there is no value for a feature. 
    By masking these missing values, the model can focus on the non-zero values and make more accurate predictions.

    Input: X_train (batch_size, seq_length, input_size) where seq_length is 48 and input_size is 82
    Output: (batch_size, seq_length, output_size) where seq_length is 48 and output_size is 82. This is the Multi-output time series forecast of all the features for all the time steps.
    '''

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(Seq2Seq, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.encoder = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout = dropout_rate)
        self.decoder = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout = dropout_rate)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def mask(self, x):
        mask = (x != 0).float()
        return x * mask

    def forward(self, x, hidden=None):
        x = self.mask(x) # masking the 0 values in the input tensor
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            hidden = (h0, c0)
        
        encoder_outputs, hidden = self.encoder(x, hidden)
        encoder_outputs = self.fc(encoder_outputs) #to change the output size of the encoder to match the input size of the decoder (82 in our case)
        decoder_outputs, _ = self.decoder(encoder_outputs, hidden)

        # Pass the final hidden state of the encoder through a linear layer to produce the final output
        out = self.fc(decoder_outputs) # (batch_size, seq_length, output_size):  The output is a 3D tensor with the predicted time series for all time steps and all the features.
        
        #out = self.fc(decoder_outputs[:, -1, :]) # (batch_size, output_size):  The output is a 2D tensor with the predicted values for all the features at the final time step.

        return out

#TODO: modify the settings to match prev model
class LSTM_Attention(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(LSTM_Attention, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout = dropout_rate, bidirectional=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.attention = torch.nn.Linear(2*hidden_size, 1)
        
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        #attention
        attention_weights = torch.nn.functional.softmax(self.attention(out), dim=1)
        out = out * attention_weights
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        # Convert the final state to our desired output shape (batch_size, output_size)
        out = self.fc(out)

        return out