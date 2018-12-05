# -*- coding: utf-8 -*-

import torch
print(torch.__version__)

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N     = 8  # batch size
D_in  = 8  # input dimension
H     = 20 # hidden dimension
D_out = 8  # output dimension

# adding the family age and sex as input parameters
x = torch.tensor([
    # molle, ebba, carin, anders, madde, erik, pappa, mamma
    [ 4,     6,    28,    33,     33,    35,   62,    63],
    [ 0,     1,    1 ,    0  ,    1,     0,    0,     1]
],device=device, dtype=dtype)

# adding the salary guesses as output parameters
y = torch.tensor([
    # molle, ebba, carin, anders, madde, erik,  pappa, mamma
    [ 0,     0,    28000, 41000,  30000, 59000, 74000, 19000]
], device=device, dtype=dtype)


# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype) # notice the H, seems we are working across all hidden layers
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    # Forward pass: compute predicted y
    h = x.mm(w1) # <-- Matrix Multiplication x(input values * w1)
    h_relu = h.clamp(min=0) # Set minimum value to 0
    y_pred = h_relu.mm(w2) # 

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item() # total loss across all 
    print(t, loss)

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