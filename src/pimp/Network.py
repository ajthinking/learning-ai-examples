import torch.nn as nn

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.l1 = nn.Linear(242, 242)
        self.l2 = nn.Linear(242, 242)
        self.l3 = nn.Linear(242, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # implement the forward pass
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred
