import torch.nn as nn
import sys

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # input data has 99 rows.
        # Each row has 242 input neurons, 1 output neurons.
        self.l1 = nn.Linear(243, 243)
        self.l2 = nn.Linear(243, 243)
        self.l3 = nn.Linear(243, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # implement the forward pass
        out1 = self.l1(x)
        
        out2 = self.l2(out1)
        y_pred = self.l3(out2)
        #print(y_pred)
        #sys.exit()        
        return y_pred
