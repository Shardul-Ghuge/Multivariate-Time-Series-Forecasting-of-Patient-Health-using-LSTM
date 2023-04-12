import torch
import random

# check if GPU is available
if(torch.cuda.is_available()):
    print('Training on GPU!')
else: 
    print('Training on CPU!')
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Base LSTM model 
class BASE_LSTM(torch.nn.Module):
    """
    A simple LSTM model for multivariate time series forecasting.
    
    The model takes an input tensor of shape (batch_size, seq_length, input_size) and passes it through an LSTM layer.
    The output of the LSTM layer is then passed through a fully connected (Linear) layer to produce the final output
    of shape (batch_size, seq_length, output_size), which represents the predicted time series for all time steps.

    Attributes:
        num_layers (int): Number of LSTM layers.
        hidden_size (int): Number of hidden units in the LSTM layers.
        lstm (nn.LSTM): LSTM layer.
        fc (nn.Linear): Fully connected (Linear) layer.

    Args:
        input_size (int): The number of input features (dimensions) of the time series.
        hidden_size (int): The number of hidden units in the LSTM layers.
        num_layers (int): The number of LSTM layers.
        output_size (int): The number of output features (dimensions) of the predicted time series.
        dropout_rate (float): The dropout rate for the LSTM layers.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(BASE_LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout = dropout_rate)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        """
        Forward pass of the LSTM model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).

        Returns:
            out (torch.Tensor): Output tensor of shape (batch_size, seq_length, output_size),
                                representing the predicted time series for all time steps.
        """
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        # Results in tensor with shape (batch_size, seq_length, output_size) representing the predicted time series for all time steps.
        out = self.fc(out)
        return out


class Seq2Seq(torch.nn.Module):
    """
    Seq2Seq model for multivariate time series forecasting.

    This model consists of an encoder and a decoder. The encoder processes the input time series and generates a hidden state representation.
    The decoder generates the output time series using the hidden states from the encoder.
    
    Args:
        input_size (int): The number of input features.
        hidden_size (int): The size of the hidden states in the LSTM layers.
        num_layers (int): The number of LSTM layers in both the encoder and the decoder.
        output_size (int): The number of output features.
        dropout_rate (float): The dropout rate for the LSTM layers.

    Input:
        x (torch.Tensor): The input tensor with shape (batch_size, seq_length, input_size).
        hidden (tuple, optional): The initial hidden and cell states for the encoder and decoder. Default: None.

    Output:
        out (torch.Tensor): The output tensor with shape (batch_size, seq_length, output_size), representing the predicted time series for all time steps and all features.

    Example:

        input_size = 82
        hidden_size = 128
        num_layers = 2
        output_size = 82
        dropout_rate = 0.1
        model = Seq2Seq(input_size, hidden_size, num_layers, output_size, dropout_rate)

        batch_size = 64
        seq_length = 48
        x = torch.randn(batch_size, seq_length, input_size)

        output = model(x)
    """
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
        #x = self.mask(x) # removed for now as I realized the model should learn to handle the sparsity in the data implicitly. Also observed that performance does not change with masking.
        if hidden is None:
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device) 
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
            hidden = (h0, c0)
        
        encoder_outputs, hidden = self.encoder(x, hidden)
        decoder_outputs, _ = self.decoder(torch.zeros_like(x), hidden)

        # Pass the final hidden state of the encoder through a linear layer to produce the final output
        out = self.fc(decoder_outputs) # (batch_size, seq_length, output_size):  The output is a 3D tensor with the predicted time series for all time steps and all the features.
        
        return out


class Seq2Seq_TeacherForcing(torch.nn.Module):
    """
    A seq2seq model with teacher forcing for multivariate time series forecasting.
    
    The model consists of an encoder and a decoder LSTM. The encoder processes the input time series and generates
    hidden states, which are then used by the decoder to produce the output time series. During training, teacher forcing
    is applied with a specified probability to help the decoder learn better.

    Attributes:
        num_layers (int): Number of LSTM layers.
        hidden_size (int): Number of hidden units in the LSTM layers.
        output_size (int): Number of output features (dimensions) of the predicted time series.
        encoder (nn.LSTM): Encoder LSTM.
        decoder (nn.LSTM): Decoder LSTM.
        fc (nn.Linear): Fully connected (Linear) layer.

    Args:
        input_size (int): The number of input features (dimensions) of the time series.
        hidden_size (int): The number of hidden units in the LSTM layers.
        num_layers (int): The number of LSTM layers.
        output_size (int): The number of output features (dimensions) of the predicted time series.
        dropout_rate (float): The dropout rate for the LSTM layers.
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(Seq2Seq_TeacherForcing, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.encoder = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.decoder = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, y, teacher_forcing_ratio=0.5):
        # Encoder
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        _, hidden = self.encoder(x, (h0, c0))

        # Decoder
        decoder_input = x[:, -1, :].unsqueeze(1)  # Use the last time step from the input as the initial decoder input
        decoder_outputs = torch.zeros(x.size(0), x.size(1), self.output_size).to(device)

        for t in range(x.size(1)):
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            out = self.fc(decoder_output)  # Transform the hidden state to the output shape
            decoder_outputs[:, t, :] = out.squeeze(1)

            # Teacher forcing: Use the ground truth as input for the next time step
            if random.random() < teacher_forcing_ratio:
                decoder_input = y[:, t, :].unsqueeze(1)
            else:
                decoder_input = out

        return decoder_outputs
    

class Attention(torch.nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: (batch_size, seq_length, hidden_size)
        # decoder_hidden: (batch_size, hidden_size)

        # Calculate the dot product between each encoder output and the decoder hidden state
        scores = torch.matmul(encoder_outputs, decoder_hidden.unsqueeze(2)).squeeze(2)  # (batch_size, seq_length)

        # Apply softmax to obtain attention weights
        attn_weights = torch.softmax(scores, dim=1)  # (batch_size, seq_length)

        # Calculate the weighted sum of the encoder outputs using the attention weights
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (batch_size, hidden_size)

        return context

class Seq2Seq_Attention(torch.nn.Module):
    """
    Seq2Seq model with attention and Teacher Forcing for multivariate time series forecasting.

    This model consists of an encoder, an attention mechanism, and a decoder.
    The encoder processes the input time series and generates a hidden state representation.
    The attention mechanism computes context vectors based on the encoder hidden states and the current decoder hidden state.
    The decoder generates the output time series using the concatenation of the context vectors and its input (which can be either ground truth or its own previous output).

    Args:
        input_size (int): The number of input features.
        hidden_size (int): The size of the hidden states in the LSTM layers.
        num_layers (int): The number of LSTM layers in both the encoder and the decoder.
        output_size (int): The number of output features.
        dropout_rate (float): The dropout rate for the LSTM layers.

    Input:
        x (torch.Tensor): The input tensor with shape (batch_size, seq_length, input_size).
        y (torch.Tensor): The ground truth tensor with shape (batch_size, seq_length, output_size).
        teacher_forcing_ratio (float, optional): The probability of using teacher forcing during training. Default: 0.5.

    Output:
        decoder_outputs (torch.Tensor): The output tensor with shape (batch_size, seq_length, output_size), representing the predicted time series for all time steps and all features.

    Example:

        input_size = 82
        hidden_size = 128
        num_layers = 2
        output_size = 82
        dropout_rate = 0.1
        model = Seq2Seq(input_size, hidden_size, num_layers, output_size, dropout_rate)

        batch_size = 64
        seq_length = 48
        x = torch.randn(batch_size, seq_length, input_size)
        y = torch.randn(batch_size, seq_length, output_size)

        output = model(x, y)
    """
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_rate):
        super(Seq2Seq_Attention, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.encoder = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.decoder = torch.nn.LSTM(input_size + hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = torch.nn.Linear(hidden_size, output_size)
        self.attention = Attention()

    def forward(self, x, y, teacher_forcing_ratio=0.5):
        # Encoder
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        encoder_outputs, hidden = self.encoder(x, (h0, c0))

        # Decoder
        decoder_input = x[:, -1, :].unsqueeze(1)  # Use the last time step from the input as the initial decoder input
        decoder_outputs = torch.zeros(x.size(0), x.size(1), self.output_size).to(device)

        for t in range(x.size(1)):
            context = self.attention(encoder_outputs, hidden[0][-1])  # Compute attention context
            decoder_input = torch.cat([decoder_input, context.unsqueeze(1)], dim=-1)  # Concatenate context with decoder input
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            out = self.fc(decoder_output)  # Transform the hidden state to the output shape
            decoder_outputs[:, t, :] = out.squeeze(1)

            # Teacher forcing: Use the ground truth as input for the next time step
            if random.random() < teacher_forcing_ratio:
                decoder_input = y[:, t, :].unsqueeze(1)
            else:
                decoder_input = out

        return decoder_outputs