import torch
from torch import nn

from sacred.config.custom_containers import ReadOnlyList

from noise_operator.config import NoNoiseConfig
from noise_operator.factory import NoiseOperatorFactory
from noise_operator.operators import NoiseOperator

cfg = {
    'ResNet18': ["B@2", "B@2", "B@2", "B@2"],
    'ResNet34': ["B@3", "B@4", "B@6", "B@3"],
    "ResNet50": ["BK@3", "BK@4", "BK@6", "BK@3"],
}

# BK for bottleneck, [1x1, 3x3, 1x1]. non-implemented

class Block(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, noise_operator=None):
        super(Block, self).__init__()
        if noise_operator is None:
            raise ValueError("noise_operator is required!")
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            noise_operator.get_noise_operator(),
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            noise_operator.get_noise_operator(),
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels, kernel_size=1, stride=stride, bias=False),
                noise_operator.get_noise_operator(),
                nn.BatchNorm2d(self.expansion*out_channels)
            )
        self.relu = nn.Sequential(
            nn.ReLU(),
            noise_operator.get_noise_operator(),
        )
        
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, noise_operator=None):
        super(Bottleneck, self).__init__()
        if noise_operator is None:
            raise ValueError("noise_operator is required!")
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_planes, planes, kernel_size=1, bias=False),
            noise_operator.get_noise_operator(),
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
            noise_operator.get_noise_operator(),
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Sequential(
            nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False),
            noise_operator.get_noise_operator(),
        )
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                noise_operator.get_noise_operator(),
                nn.BatchNorm2d(self.expansion*planes)
            )
        self.relu = nn.Sequential(
            nn.ReLU(),
            noise_operator.get_noise_operator(),
        )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out = self.relu(out)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    """
    
    """
    def __init__(self,
                 conf_name='ResNet18',
                 input_shape=(1, 28, 28),
                 default_noise_config=NoNoiseConfig(),
                 layer_wise_noise_config=dict(),
                 num_classes=10,
                 ):
        super().__init__()
        # Makes some noise xD
        self._noise_factory = NoiseOperatorFactory(default_noise_config, layer_wise_config=layer_wise_noise_config)
        
        # build network
        block = Block
        num_blocks = []
        for block_info in cfg[conf_name]:
            if block_info.startswith("BK"):
                block = Bottleneck
                num_blocks.append(int(block_info.split('@')[-1]))
            elif block_info.startswith("B"):
                num_blocks.append(int(block_info.split('@')[-1]))
        
        self.in_planes = 64
        in_channels = input_shape[0]
        temp_size = input_shape[1]
        if temp_size < 224: #temp_size == 64: #Tiny ImageNet
            # the original experiments on CIFAR10 using this large kernel size
            self.conv1 = nn.Sequential(
                self._noise_factory.get_noise_operator(),
                #nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3), # original 7,2,3, recommend 3,1,1. Might be the reason that resnet can only achieve 88%
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1), 
                self._noise_factory.get_noise_operator()
            )
        else: # to keep consistent with previous experiments
            self.conv1 = nn.Sequential(
                self._noise_factory.get_noise_operator(),
                nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3), # original 7,2,3, recommend 3,1,1. Might be the reason that resnet can only achieve 88%
                #nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1), 
                self._noise_factory.get_noise_operator()
            )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.Sequential(
            nn.ReLU(),
            self._noise_factory.get_noise_operator(),
        )
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._noise_factory.get_noise_operator(),
        )
        
        self.layer1 = self._make_layers(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layers(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layers(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layers(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            self._noise_factory.get_noise_operator(),
        )
        self.fc = nn.Sequential(
            nn.Linear(512*block.expansion, num_classes),
            self._noise_factory.get_noise_operator(),
        )
        num_noise_layers = self._noise_factory._layer_counter
        print(f"Created the following number of noise layers / operators: {num_noise_layers}")
        if self._noise_factory.check_for_unused_configs():
            raise ValueError(f"A noise setting for a layer not contained in the network was requested. "
                             f"This is likely due to an incorrect configuration. "
                             f"The built network has {num_noise_layers} noise layers, "
                             f"but layer wise configurations were requested for the following layer indices: "
                             f"{layer_wise_noise_config.keys()}")

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    def _make_layers(self, block, out_channels, num_block, stride=1,):
        strides = [stride] + [1]*(num_block-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, out_channels, stride, noise_operator=self._noise_factory))
            self.in_planes = out_channels * block.expansion
        return nn.Sequential(*layers)

