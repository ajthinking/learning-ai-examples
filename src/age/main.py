# -*- coding: utf-8 -*-

import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

D_in  = 1  # input dimension
H     = 50 # hidden dimension
D_out = 1  # output dimension

# adding the family age input parameters (normalize with max age=35)
x = torch.tensor([
    [4/35],
    [6/35],
    [28/35],
    [33/35],
    [35/35],        
],device=device, dtype=dtype)

# adding the age in days as output parameters
y = torch.tensor([
    [4*365],
    [6*365],
    [28*365],
    [33*365],
    [35*365],
], device=device, dtype=dtype)


# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype) # notice the H, seems we are working across all hidden layers
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

latest_loss = 0

learning_rate = 1e-5
for epoch in range(1000):
    # Forward pass: compute predicted y
    h = x.mm(w1) # <-- Matrix Multiplication x(input values * w1)
    h_relu = h.clamp(min=0) # Set minimum value to 0
    y_pred = h_relu.mm(w2) #



    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item() # total loss across all 
    print(epoch, loss)

    # check for convergence
    if latest_loss == loss:
        break
    
    latest_loss = loss


    # Backprop to compute gradients of w1 and w2 with respect to loss
    grad_y_pred = 2.0 * (y_pred - y) # get the slope from the y difference (positive or negative times two)
    grad_w2 = h_relu.t().mm(grad_y_pred) # t() transposes dimensions 0 and 1
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

print("Suddenly the unknown sibling Jerry (age x) joins your family. What are his number of survived days (y)?")
h = torch.tensor([
    [45/35], # Jerry
],device=device, dtype=dtype).mm(w1) # <-- Matrix Multiplication x(input values * w1)
h_relu = h.clamp(min=0) # Set minimum value to 0
y_pred = h_relu.mm(w2) #
print(
    y_pred.flatten().item()
)