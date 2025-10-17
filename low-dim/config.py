from argparse import ArgumentParser
import random
import time
from typing import Literal, TypeAlias, NoReturn, List, Union

def set_random_seed(seed: int):
    import torch
    import torch.cuda as cuda
    import torch.backends as backends
    import numpy as np

    torch.manual_seed(seed)
    cuda.manual_seed_all(seed)
    backends.cudnn.deterministic = True
    backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)



def get_arg_parser():
    par = ArgumentParser()

    # ----------------------------   Universal   ----------------------------
    
    par.add_argument ("--seed"      , type = int    , default = 2333)
    par.add_argument ("--device"    , type = str    , default = "cuda:0"    , choices=["cuda:0", "cpu"])
    par.add_argument ("--info" 		, type = str    , default = "")

    # ----------------------------    Model & Data       ----------------------------

    par.add_argument ("--model"       , type = str    , default = "mlp" )
    par.add_argument ("--hidden_size" , type = int    , default = 512 )
    par.add_argument ("--num_epoch"   , type = int    , default = 10 )
    par.add_argument ("--data"        , type = str    , default = "teacher", choices = ["teacher", "teacher_3d"])

    par.add_argument("--num_data"  , type = int         , default = 5000)
    par.add_argument("--train_prop", type = float       , default = 0.7 )
    par.add_argument("--alt"       , action="store_true", default = False)


    # ----------------------------    Optimizer       ----------------------------
    par.add_argument ("--optimizer" , type = str  , default = "SGD" ) 
    par.add_argument ("--bs" , type = int         , default = 128   )
    par.add_argument ("--lr" , type = float       , default = 0.001  )
    par.add_argument ("--wd" , type = float       , default = 0.0  )

    return par
