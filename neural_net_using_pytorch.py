import torch
import torch.nn as nn
import torch.nn.functional as F


input_size, hidden_size, output_size, batch_size = 10, 15, 1, 10
x = torch.randn(batch_size, input_size)
y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])


# building a model using class
class SimpleNeuralNet(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        
        super(SimpleNeuralNet, self).__init__()
        
        # define the parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # define the network
        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.output_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x):
        
        # define forward propagation
        fc1_out = self.fc1(x)
        non_linear_out = self.relu(fc1_out)
        fc2_out = self.fc2(non_linear_out)
        out = self.sigmoid(fc2_out)
        return out

# create the model object
model = SimpleNeuralNet(input_size, hidden_size, output_size)

# define the loss function
criterion = nn.MSELoss()

# define the optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#training the model over specified epochs
for epoch in range(50):
    
    #compute forward prop
    y_pred = model.forward(x)
    
    # compute and print loss
    loss = criterion(y_pred, y)
    print('[info] epoch: {}, loss: {}'.format(epoch, loss))

    # reset gradient to zero
    optimizer.zero_grad()

    #backward propagation
    loss.backward()

    # update the paramters
    optimizer.step()


    
        



