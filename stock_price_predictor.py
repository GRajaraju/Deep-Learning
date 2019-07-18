# import the requried libraries
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# import the data
stock_raw_data = pd.read_csv('/home/raj/Downloads/GOOG.csv')
data = stock_raw_data.iloc[:, 1:2].values
data = (data - np.min(data)) / (np.max(data) - np.min(data))
x = data[:-1]
y = data[1:]
print('x shape: ', x.shape)
# define sequence length, time steps
sequence_length = len(data) - 1
time_steps = np.linspace(0, 1, sequence_length+1)

'''
# plot the data
plt.plot(time_steps[1:], x, 'r.', label='original stock, x')
plt.plot(time_steps[1:], y, 'b.', label='predicted stock, y')
plt.show()
'''

# define the sequence model - RNN
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_data, hidden_state):
        output, h_s = self.rnn(input_data, hidden_state)
        output = output.view(-1, self.hidden_size)
        out = self.fc(output)
        return out, h_s


input_size = 1
output_size = 1
hidden_size = 100
num_layers = 1
batch_size = 1 
input_data = x.reshape(batch_size, sequence_length, input_size)

# create RNN object
rnn = RNN(input_size, output_size, hidden_size, num_layers)
input_data = torch.Tensor(input_data)
y = torch.Tensor(y)
hidden_state = None

# define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.01)

# train the RNN
def training(rnn, epochs):
    # initialize the hidden state
    hidden = None

    for epoch in range(epochs):
        prediction, hidden = rnn(input_data, hidden)
        hidden = hidden.data

        # calculate the loss
        loss = criterion(prediction, y)
        # set zero gradients
        optimizer.zero_grad()
        # perform backprop and update weights
        loss.backward()
        optimizer.step()

        # print training error
        if epoch % 10 == 0:
            print(loss.item())
    
    print('prediction shape: ', prediction.shape)
    plt.plot(time_steps[1:], x, 'r.', label='original sequence, x')
    plt.plot(time_steps[1:], prediction.data.numpy().flatten(), 'b.', 
                    label='predicted sequence, y')
    plt.show()
    
    return rnn

epochs = 50
training(rnn, epochs)


