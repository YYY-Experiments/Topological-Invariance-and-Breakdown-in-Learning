from hessian_eigenthings import compute_hessian_eigenthings
from torch.nn.functional import cross_entropy

def calc_sharpness(model, dataloader):
    
    model.eval()
    eigenvals, _ = compute_hessian_eigenthings(
        model, 
        dataloader, 
        cross_entropy, 
        num_eigenthings=1,
        mode="power_iter",
        use_gpu=True , 
        full_dataset=False, 
    )

    max_eigenvalue = eigenvals[0]
    return max_eigenvalue