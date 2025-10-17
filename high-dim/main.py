import numpy as np
import os, sys
from pprint import pformat

import torch as tc
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from argparse import Namespace
import random
import pickle

from dataloaders import dataloaders
from models import model_getters 
from optimizers import get_optimizer 
from train_and_test import train , test
from config import get_arg_parser, set_random_seed
from topology import betti_numbers
from sharpness import calc_sharpness

def main(C: Namespace, exp_id:str):

    device              =  str  (C.device)          
    data_name           =  str  (C.data)     
    num_epoch           =  int  (C.num_epoch)
    model_name          =  str  (C.model)    
    optimizer_name      =  str  (C.optimizer)
    bs                  =  int  (C.bs)       
    lr                  =  float(C.lr)       
    wd                  =  float(C.wd)       
    hidden_size         =  int  (C.hidden_size)

    dataloader  = dataloaders[data_name]
    trainset , testset = dataloader()
    trainloader = DataLoader(trainset, batch_size = bs, shuffle = True)
    testloader  = DataLoader(testset , batch_size = bs, shuffle = False)
    
    model_getter = model_getters[ model_name ]
    input_size   = trainset.data[0].view(-1).shape[0]
    output_size  = len(trainset.classes)
    model = model_getter(input_size, hidden_size, output_size)
    model.to(device)

    optim = get_optimizer(optimizer_name, model, lr, wd)
    
    info = {
        key_name: [] for key_name in [
            "train loss", "train acc", "test loss", "test acc",
            "b0", "b1", "b2", "Kinv",
        ]
    }
    train_loss , train_acc = 0.0 , 0.0
    for epoch_idx in range(num_epoch):

        test_loss , test_acc = test(epoch_idx , model, testloader, device)

        neurons = model.get_neurons()
        b0, b1, b2 = betti_numbers(neurons)
        K = calc_sharpness(model, testloader)

        desc = f"[{epoch_idx}] train: {train_loss:.4f} | {train_acc:.2f}%, test: {test_loss:.4f} | {test_acc:.2f}%, b0 = {b0}, b1 = {b1}, b2 = {b2}, 1/K = {1/K:.4f}"
        print(desc)

        info["train loss"].append(train_loss)
        info["train acc" ].append(train_acc )
        info["test loss" ].append(test_loss )
        info["test acc"  ].append(test_acc  )
        info["b0"        ].append(b0        )
        info["b1"        ].append(b1        )
        info["b2"        ].append(b2        )
        info["Kinv"      ].append(1/K       )

        # train & test
        model , train_loss , train_acc = train(
            epoch_idx , 
            model , 
            trainloader , 
            optim, 
            device, 
        )

    test_loss , test_acc = test(num_epoch , model , testloader, device)
    print (f"final test_loss = {test_loss:.4f}, final test_acc = {test_acc:.2f}")

    save_path = f"./results/{exp_id}.pkl"
    os.makedirs( os.path.dirname(save_path) , exist_ok = True)
    with open(save_path, "wb") as fil:
        pickle.dump(info , fil)
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