from Network import Network
from DrawingsDataset import DrawingsDataset
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import sys
from torch.autograd import Variable

drawings = DrawingsDataset('/home/anders/Code/')
train_loader = DataLoader(dataset=drawings,
    batch_size=1000,
    shuffle=True,
    num_workers=2)

# our model
network = Network()


# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = nn.MSELoss()
optimizer = optim.SGD(network.parameters(), lr=0.1)


# Updated training loop
for epoch in range(1000):
    for i, data in enumerate(train_loader, 0):
        #print(data.size())
        #sys.exit()
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

        # Forward pass: Compute predicted y by passing x to the model
        y_pred = network(inputs)
        # Compute and print loss
        loss = criterion(y_pred, labels)
        print(epoch, i, loss.data[0])

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


print("finished!")