import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.utils import standardize
from gpytorch.mlls import ExactMarginalLogLikelihood
# torch.device()
# import torch

print("Is CUDA supported by this system?",torch.cuda.is_available())
# print(torch.version.cuda)

# Storing ID of current CUDA device
# cuda_id = torch.cuda.current_device()
# print(torch.cuda.current_device())
	
# print(f"Name of current CUDA device:
# 	{torch.cuda.get_device_name(cuda_id)}")

train_X = torch.rand(10, 2)
Y = 1 - torch.norm(train_X - 0.5, dim=-1, keepdim=True)
Y = Y + 0.1 * torch.randn_like(Y)  # add some noise
train_Y = standardize(Y)

gp = SingleTaskGP(train_X, train_Y)
mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)