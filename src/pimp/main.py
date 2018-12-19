from Network import Network
from DrawingsDataset import DrawingsDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import sys
from torch.autograd import Variable

print('******************************************************')
print("\nTraining...\n")

train_loader = DataLoader(
    dataset=DrawingsDataset('/home/anders/Code/', train=True),
    batch_size=1000,
    shuffle=True,
    num_workers=2)

test_loader = DataLoader(
    dataset=DrawingsDataset('/home/anders/Code/', test=True),
    shuffle=True,
    num_workers=2)    

# our model
network = Network()

# Mean Square Error Loss
criterion = nn.MSELoss()
# Stochasctic gradient descent
optimizer = optim.SGD(network.parameters(), lr=0.01)

# training loop
print("epoch", '\t', "iteration", '\t', "loss")
for epoch in range(10000):
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, output = data

        # wrap them in Variable
        inputs, output = Variable(inputs), Variable(output)

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = network(inputs)
        # Compute and print loss
        loss = criterion(y_pred, output)
        
        print(epoch, '\t', i, '\t', loss.item())

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

print("\nTesting...\n")
total_test_loss = 0
for i, data in enumerate(test_loader, 0):
    inputs, output = data
    prediction = int(network(inputs).data.item() * 9724) 
    actual = int(output.item() * 9724) 
    loss = abs(prediction - actual)
    total_test_loss += loss
    print("Drawing", i, "predicted to ", prediction, "downloads. Error = ", loss)

print("\nfinished testing with a average loss of", int(total_test_loss/10) ,"downloads per drawing\n")

#hour_var = Variable(torch.Tensor([[1.0]]))
#print("predict 1 hour ", 1.0, network(hour_var).data[0][0] > 0.5)
#hour_var = Variable(torch.Tensor([[7.0]]))
#print("predict 7 hours", 7.0, network(hour_var).data[0][0] > 0.5)