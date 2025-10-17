import torch as tc
import torch.nn as nn

def get_optimizer(
        optimizer_name: str, 
        model: nn.Module, 
        lr: float, 
        wd: float = 0, 
    ):

    optim = None

    if optimizer_name == "SGD":
        optim = tc.optim.SGD(model.parameters() , lr = lr , weight_decay = wd)

    if optimizer_name == "adam":
        optim = tc.optim.Adam(model.parameters() , lr = lr , weight_decay = wd)

    if optim is None:
        raise ValueError("Bad optimizer name.")

    return optim


