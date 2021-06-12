import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm
from typing import Type, Any, Callable, Union, List, Optional
import torch
from torch import Tensor
import torch.nn as nn
from .utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional


class BasicBlock(nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                  downsample: Optional[nn.Module] = None, groups: int = 1, 
                  base_width: int = 64, dilation: int = 1, 
                  norm_layer: Optional[Callable[..., nn.Module]] = None) -> None:
        super(self).__init__()
        self.expansion = 1
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64: ## 牽涉到 group, channel等設定
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
            
            
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84,  10)
        
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
        
