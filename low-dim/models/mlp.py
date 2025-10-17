import torch as tc
import torch.nn.functional as F
import torch.nn as nn
from .initializations import initialize_2d, initialize_3d, alt_initialize_2d


class MLP(nn.Module):
    def __init__( 
        self, input_size, hidden_size , num_classes, alt , 
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.alt = alt

        self.ln  = nn.Linear (input_size , hidden_size, bias = False)
        self.out = nn.Linear(hidden_size , num_classes, bias = False)

        self.initialize_weight()


    def initialize_weight(self):
        H = self.hidden_size
        D = self.input_size

        if D == 1:
            print ("2d initialization")
            xs, ys = initialize_2d(H)
            if self.alt:
                xs, ys = alt_initialize_2d(H, radius_top=1.0, radius_bot=1.0, gap=2.4)
            for i in range(H):
                x,y = xs[i], ys[i]
                x = x * 0.1 - 2
                self.ln.weight.data[i,0] = x
                self.out.weight.data[0,i] = y
        elif D == 2:
            print ("3d initialization")
            xs, ys, zs = initialize_3d(H)
            for i in range(H):
                x,y,z = xs[i], ys[i], zs[i]
                x = x * 0.1 - 2
                
                self.ln.weight.data[i,0] = x
                self.ln.weight.data[i,1] = y
                self.out.weight.data[0,i] = z
        else:
            raise NotImplementedError("Unsupported input dimension")


    def log_weight(self):

        W1 = self.ln.weight.data  # (hidden_size, input_size)
        W2 = self.out.weight.data # (num_classes, hidden_size)

        return [ [float(W1[i,j]) for j in range(len(W1[i]))] + [float(W2[j,i]) for j in range(len(W2[:,i]))] for i in range(self.hidden_size) ]

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.sigmoid(self.ln(x))
        h = x
        x = self.out(x)
        
        return {
            "pred": x , 
            "hidden": h , 
        } 

def get_mlp(input_size, hidden_size, num_class, alt):
    return MLP(input_size, hidden_size, num_class, alt)

