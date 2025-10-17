from .mnist import load_data as mnist_load_data

dataloaders = {
	"mnist"		: mnist_load_data,
}