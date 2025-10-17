import torch as tc
import torch.nn.functional as F
import torch.nn as nn
from .initializations import initialize_S2


class MLP(nn.Module):
    def __init__( 
        self, input_size, hidden_size , num_classes , 
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.ln  = nn.Linear (input_size , hidden_size, bias = False)
        self.out = nn.Linear(hidden_size , num_classes, bias = False)

        self.initialize_weight()

    def get_neurons(self):
        W1 = self.ln.weight.data  # (hidden_size, input_size)
        W2 = self.out.weight.data.t() # (hidden_size, num_classes)

        return tc.cat([W1, W2], dim = 1) # (hidden_size, input_size + num_classes)

    def get_neuron_params(self):
        return [self.ln.weight, self.out.weight]
    
    def initialize_weight(self):
        W1 = self.ln.weight.data # (hidden_size, input_size)
        W2 = self.out.weight.data # (num_classes, hidden_size)
        device = W1.device

        n,d = W1.size(0), W1.size(1) + W2.size(0)
        X = initialize_S2(n,d)

        W1.copy_( X[:, :W1.size(1)].contiguous().to(device) )
        W2.copy_( X[:, W1.size(1):].t().contiguous().to(device) )

    def forward(self, x: tc.Tensor):
        '''
            x: (bs,3,32,32)
        '''
        x = x.view(-1, self.input_size)
        x = F.sigmoid(self.ln(x))
        x = self.out(x)
        
        return x

def get_mlp(input_size, hidden_size, num_class):
    return MLP(input_size, hidden_size, num_class)
