import enum
import torch as tc
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pdb
import copy
from typing import Union, Callable

def train(
	epoch_idx: int , 
	model: nn.Module , 
	dataloader: DataLoader, 
	optim: tc.optim.Optimizer, 
	device: str = "cuda:0", 
	hook: Callable[[nn.Module], None] | None = None,
):
	model = model.train()

	tot_loss = 0
	tot_sample = 0
	tot_hit = 0
	pbar = tqdm( enumerate(dataloader) , postfix = "training %dth epoch" % epoch_idx)
	for batch_idx , data in pbar:
		input , labels = data
		input  = input .to(device)
		labels = labels.to(device)

		# get output
		preds = model(input)

		loss = F.cross_entropy(preds , labels)

		# backward
		optim.zero_grad()
		loss.backward()
		optim.step()

		if hook is not None:
			hook(model)

		# record statistics
		num_sample = len(labels)

		inc_loss = float(loss) * num_sample
		inc_hit = int( (tc.max(preds , -1)[1] == labels).long().sum() )

		tot_loss = tot_loss + inc_loss
		tot_hit = tot_hit + inc_hit
		tot_sample = tot_sample + num_sample

		desc = "loss = %.4f , acc = %.2f" % (inc_loss / num_sample , inc_hit / num_sample * 100)
		pbar.set_description(desc)

	loss = tot_loss / tot_sample
	acc = tot_hit / tot_sample

	return model , loss , acc * 100

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
	tot_hit = 0
	pbar = tqdm( enumerate(dataloader) , postfix = "Testing %dth epoch" % epoch_idx)
	for batch_idx , data in pbar:
		input , labels = data
		input = input.to(device)
		labels = labels.to(device)
		preds = model(input)

		loss = F.cross_entropy(preds , labels)

		# record statistics
		num_sample = len(labels)

		inc_loss = float(loss) * num_sample
		inc_hit = int( (tc.max(preds , -1)[1] == labels).long().sum() )

		tot_loss = tot_loss + inc_loss
		tot_hit = tot_hit + inc_hit
		tot_sample = tot_sample + num_sample

		desc = "loss = %.4f , acc = %.2f" % (inc_loss / num_sample , inc_hit / num_sample * 100)
		pbar.set_description(desc)

	loss = tot_loss / tot_sample
	acc = tot_hit / tot_sample

	return loss , acc * 100