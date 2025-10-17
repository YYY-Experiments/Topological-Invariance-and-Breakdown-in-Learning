import torchvision.transforms as transforms
from torchvision.datasets import MNIST

def load_data():
	transform_train = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,)),
	])

	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.1307,), (0.3081,)),
	])

	trainset = MNIST(root='./data', train=True,download=True, transform=transform_train)
	testset  = MNIST(root='./data', train=False,download=True, transform=transform_test)

	return trainset , testset
