
import pickle 
import os

def get_data(exp_id, key):
    file_path = f"./results/{exp_id}.pkl"
    with open(file_path, "rb") as f:
        info = pickle.load(f)
    return info[key]
