import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Union, Callable

def train(
	epoch_idx: int , 
	model: nn.Module , 
	dataloader: DataLoader, 
	optim: tc.optim.Optimizer, 
	device: str = "cuda:0", 
	weight_logger: Callable = None,
):
	model = model.train()

	tot_loss = 0
	tot_sample = 0
	pbar = tqdm( enumerate(dataloader) , postfix = "training %dth epoch" % epoch_idx)
	for batch_idx , data in pbar:
		input , labels = data
		input  = input .to(device)
		labels = labels.to(device)

		# get output
		output = model(input)
		preds = output["pred"]
		
		loss = F.mse_loss(preds , labels)

		# backward
		optim.zero_grad()
		loss.backward()
		optim.step()

		if weight_logger is not None:
			weight_logger(model)

		# record statistics
		num_sample = len(labels)

		inc_loss = float(loss) * num_sample
		tot_loss = tot_loss + inc_loss
		tot_sample = tot_sample + num_sample

		desc = "loss = %.4f" % (inc_loss / num_sample)
		pbar.set_description(desc)

	loss = tot_loss / tot_sample

	return model , loss

@tc.no_grad()
def test(
	epoch_idx: int , 
	model: nn.Module , 
	dataloader: DataLoader, 
	device: str = "cuda:0", 
):
	model = model.eval()

	tot_loss = 0
	tot_sample = 0
	pbar = tqdm( enumerate(dataloader) , postfix = "Testing %dth epoch" % epoch_idx)
	for batch_idx , data in pbar:
		input , labels = data
		input = input.to(device)
		labels = labels.to(device)
		output = model(input)

		preds = output["pred"]

		loss = F.mse_loss(preds , labels)

		# record statistics
		num_sample = len(labels)

		inc_loss = float(loss) * num_sample

		tot_loss = tot_loss + inc_loss
		tot_sample = tot_sample + num_sample

		desc = "loss = %.4f" % (inc_loss / num_sample)
		pbar.set_description(desc)

	loss = tot_loss / tot_sample

	return loss
