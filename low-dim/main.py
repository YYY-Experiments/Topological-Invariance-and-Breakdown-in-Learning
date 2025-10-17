import numpy as np
import sys
from pprint import pformat
from argparse import Namespace
import torch as tc
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import random 
import os
import pickle

from dataloaders import dataloaders
from models import model_getters 
from optimizers import get_optimizer 
from train_and_test import train , test
from config import get_arg_parser, set_random_seed

_logged_weights = []
def make_weight_logger():
    def weight_logger(model: nn.Module):
        weights = model.log_weight()
        _logged_weights.append(weights)
    return weight_logger

def main(C: Namespace, exp_id: str):
    # get config
    device              =  str  (C.device)          
    data_name           =  str  (C.data)     
    num_epoch           =  int  (C.num_epoch)
    model_name          =  str  (C.model)    
    optimizer_name      =  str  (C.optimizer)
    bs                  =  int  (C.bs)       
    lr                  =  float(C.lr)       
    wd                  =  float(C.wd)       
    hidden_size         =  int  (C.hidden_size)
    num_data            =  int  (C.num_data)
    train_prop          =  float(C.train_prop)
    alt                 =  bool(C.alt)

    # get dataloader
    dataloader  = dataloaders[data_name]
    trainset , testset = dataloader(num_data, train_prop)
    trainloader = DataLoader(trainset, batch_size = bs, shuffle = True)
    testloader  = DataLoader(testset , batch_size = bs, shuffle = False)
    
    # get model
    model_getter = model_getters[ model_name ]
    input_size   = trainset.data[0].view(-1).shape[0]
    output_size  = len(trainset.classes)
    model = model_getter(input_size, hidden_size, output_size, alt = alt)
    model.to(device)

    # get optimizer
    optim = get_optimizer(optimizer_name, model, lr, wd)
    
    # start training
    weight_logger = make_weight_logger()
    for epoch_idx in range(num_epoch):

        # train & test
        model , train_loss = train(
            epoch_idx , 
            model , 
            trainloader , 
            optim, 
            device, 
            weight_logger = weight_logger
        )
        test_loss = test(epoch_idx , model, testloader, device)

        # 记录信息 info
        desc = "epoch_id = %d, train loss = %4f, test loss = %4f" % (
            epoch_idx, train_loss, test_loss
        )

        print(desc)
        
    test_loss = test(num_epoch , model , testloader, device)
    print ("final test_loss = %.4f" % (test_loss))

    logged_weights = tc.Tensor(_logged_weights)
    print("logged_weights.shape = %s" % str(logged_weights.shape))

    save_path = f"./results/{exp_id}.pkl"
    os.makedirs( os.path.dirname(save_path) , exist_ok = True)
    with open(save_path, "wb") as fil:
        pickle.dump({
            "logged_weights": logged_weights,
        } , fil)
    print(f"results saved to {save_path}")
    return model

def start():

    exp_id = str(random.randint(233333, 23333333)) 
    C = get_arg_parser().parse_args()
    
    print(" ".join(sys.argv))
    print(pformat(C))
    print(f"Experiment ID is {exp_id}")
    print("-" * 50)

    if C.seed > 0:
        set_random_seed(C.seed)
        print (f"Seed set to {C.seed}.")
    
    print (f"start train.")
    main(C, exp_id)
    print (f"end train.")

    print("-" * 50)
    print("everything end.")

if __name__ == "__main__":

    start()