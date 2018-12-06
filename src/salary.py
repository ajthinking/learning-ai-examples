# -*- coding: utf-8 -*-

import torch

dtype = torch.float
device = torch.device("cpu")
# device = torch.device("cuda:0") # Uncomment this to run on GPU

D_in  = 3  # input dimension
H     = 20 # hidden dimension
D_out = 1  # output dimension

# adding the family age and sex as input parameters
x = torch.tensor([
    [4. /63,  0., 0], # molle
    [6. /63,  1., 0], # ebba
    [28./63, 1., 0],  # carin
    [33./63, 0., 1],  # anders
    [33./63, 1., 0],  # madde
    [35./63, 0., 1],  # erik
    [62./63, 0., 1],  # pappa
    [63./63, 1., 0]   # mamma
],device=device, dtype=dtype)

# adding the salary guesses as output parameters
y = torch.tensor([
    [0.    /75000], # molle
    [0.    /75000], # ebba
    [26000./75000], # carin
    [41000./75000], # anders
    [29000./75000], # madde
    [62000./75000], # erik
    [75000./75000], # pappa
    [19000./75000]  # mamma
], device=device, dtype=dtype)


# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype) # notice the H, seems we are working across all hidden layers
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

latest = 0

learning_rate = 1e-6
for t in range(100000):
    # Forward pass: compute predicted y
    h = x.mm(w1) # <-- Matrix Multiplication x(input values * w1)
    h_relu = h.clamp(min=0) # Set minimum value to 0
    y_pred = h_relu.mm(w2) #

    latest = y_pred 

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

print("Suddenly the unknown siblings Jerry and Jenny joins the family. What are their salaries?")
h = torch.tensor([
    [64. /63,  0., 1], # Jerry
    [64. /63,  1., 0]  # Jenny
],device=device, dtype=dtype).mm(w1) # <-- Matrix Multiplication x(input values * w1)
h_relu = h.clamp(min=0) # Set minimum value to 0
y_pred = h_relu.mm(w2) #
print(y_pred * 75000)