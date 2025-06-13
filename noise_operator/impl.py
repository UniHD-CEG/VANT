import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass

##############################################################################
# define noises as transformations
# https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
##############################################################################
class AddGaussianNoise(nn.Module):
    def __init__(self, mean=0., std=1., requires_grad=False, normalize_to_input_size=False):
        super().__init__()
        if isinstance(std, float):
            std = torch.Tensor([std])
        else:
            std = torch.Tensor(std)
        self.std = nn.Parameter(std, requires_grad = requires_grad)
        self.mean = mean
        self.normalize_to_input_size = normalize_to_input_size

    def forward(self, tensor):
        device = tensor.get_device()
        if device == -1:
            device = 'cpu'
        # [1:].shape makes sure to ignore the batch dimension
        rescaler = (1. / np.prod(tensor.shape[1:])) if self.normalize_to_input_size else 1.
        #return tensor + torch.normal(self.mean, self.std * rescaler, tensor.size(), device=device)
        if len(tensor.shape) == 4:
            self.std.data = self.std.data.view(-1, 1,1,1)
        elif len(tensor.shape) == 2:
            self.std.data = self.std.data.view(-1, 1)
        noise = torch.abs(self.std) * rescaler * torch.randn(tensor.size(), device=device)
        return tensor + noise

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class MultiplyGaussianNoise(nn.Module):
    def __init__(self, mean=0., std=1., requires_grad=False, normalize_to_input_size=False):
        super().__init__()
        self.std = nn.Parameter(torch.Tensor([std]), requires_grad = requires_grad)
        self.mean = mean
        self.normalize_to_input_size = normalize_to_input_size

    def forward(self, tensor):
        device = tensor.get_device()
        if device == -1:
            device = 'cpu'
        # [1:].shape makes sure to ignore the batch dimension
        rescaler = (1./np.prod(tensor.shape[1:])) if self.normalize_to_input_size else 1.
        #return tensor * torch.normal(self.mean, self.std * rescaler, tensor.size(), device=device)
        noise = torch.abs(self.std) * rescaler * torch.randn(tensor.size(), device=device)
        return tensor * noise

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


@dataclass
class CombinedGaussianNoise(object):
    #: Order of operations
    FirstMulThenAdd: bool = True
    #: Gauss mean value for the multiplier part
    GaussMeanMul: float = 1.0
    #: Gauss standard-deviation for the multiplier part
    GaussStdMul: float = 1.0
    #: Gauss mean value for the adder part
    GaussMeanAdd: float = 0.0
    #: Gauss standard-deviation for the adder part
    GaussStdAdd: float = 1.0
    #: Weather the noise should be normalized to the number of parameters at the input tensor
    normalize_to_input_size: bool = False

    def __call__(self, tensor):
        device = tensor.get_device()
        if device == -1:
            device = 'cpu'
        # [1:].shape makes sure to ignore the batch dimension
        rescaler = (1. / np.prod(tensor.shape[1:])) if self.normalize_to_input_size else 1.
        if self.FirstMulThenAdd:
            out = tensor * torch.normal(self.GaussMeanMul, self.GaussStdMul * rescaler, tensor.size(), device=device)
            out += torch.normal(self.GaussMeanAdd, self.GaussStdAdd * rescaler, tensor.size(), device=device)
        else:
            out = tensor + torch.normal(self.GaussMeanAdd, self.GaussStdAdd * rescaler, tensor.size(), device=device)
            out *= torch.normal(self.GaussMeanMul, self.GaussStdMul * rescaler, tensor.size(), device=device)
        return out
