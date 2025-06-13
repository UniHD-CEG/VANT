import torch
from torch import nn
from brevitas import nn as qnn

from sacred.config.custom_containers import ReadOnlyList

from noise_operator.config import NoNoiseConfig
from noise_operator.factory import NoiseOperatorFactory
from noise_operator.operators import NoiseOperator

from models_noisy.util import *

cfg = {
    'ResNet18': ["B@2", "B@2", "B@2", "B@2"],
    'ResNet34': ["B@3", "B@4", "B@6", "B@3"],
    "ResNet50": ["BK@3", "BK@4", "BK@6", "BK@3"],
}
# BK for bottleneck, [1x1, 3x3, 1x1]. non-implemented

min_max_dict = {
    4: (8, 15),
    8: (128, 256),
    16: (32768, 65536)
}

class BlockQAT(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, noise_operator=None, bit_width=8, scale=None):
        super(BlockQAT, self).__init__()
        if noise_operator is None:
            raise ValueError("noise_operator is required!")
        self.conv1 = nn.Sequential(
            #nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            qnn.QuantConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, weight_bit_width=bit_width, bias=True, weight_quant=CommonIntWeightPerChannelQuant,),
            noise_operator.get_noise_operator(),
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Sequential(
            #nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            qnn.QuantConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, weight_bit_width=bit_width, bias=True, weight_quant=CommonIntWeightPerChannelQuant,),
            noise_operator.get_noise_operator(),
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        if scale is not None:
            max_uint = min_max_dict[bit_width][1] * scale
            max_int = min_max_dict[bit_width][0] * scale
            self.relu = nn.Sequential(
                qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=False, 
                                    act_quant=ConstScaledActQuantUint, max_val=max_uint),
                noise_operator.get_noise_operator()
            )
        else:
            self.relu = nn.Sequential(
                qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=False)  ,
                noise_operator.get_noise_operator()
            )
        
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class ResNetQAT(nn.Module):
    """
    
    """
    def __init__(self,
                 conf_name='ResNet18',
                 input_shape=(1, 28, 28),
                 default_noise_config=NoNoiseConfig(),
                 layer_wise_noise_config=dict(),
                 num_classes=10,
                 bit_width=8,
                 scale = None,
                 ):
        super().__init__()
        # Makes some noise xD
        self._noise_factory = NoiseOperatorFactory(default_noise_config, layer_wise_config=layer_wise_noise_config)
        
        # build network
        num_blocks = []
        for block_info in cfg[conf_name]:
            if block_info.startswith("BK"):
                raise NotImplementedError
            elif block_info.startswith("B"):
                num_blocks.append(int(block_info.split('@')[-1]))
        
        in_channels = input_shape[0]
        self.conv1 = nn.Sequential(
            self._noise_factory.get_noise_operator(),
            #nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            qnn.QuantConv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, weight_bit_width=bit_width, bias=True, weight_quant=CommonIntWeightPerChannelQuant,),
            self._noise_factory.get_noise_operator()
        )
        self.bn1 = nn.BatchNorm2d(64)
        if scale is not None:
            max_uint = min_max_dict[bit_width][1] * scale
            max_int = min_max_dict[bit_width][0] * scale
            self.relu = nn.Sequential(
                qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=False, 
                                    act_quant=ConstScaledActQuantUint, max_val=max_uint),
                self._noise_factory.get_noise_operator()
            )
        else:
            self.relu = nn.Sequential(
                qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=False),
                self._noise_factory.get_noise_operator()
            )
        self.maxpool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._noise_factory.get_noise_operator(),
        )
        
        self.layer1 = self._make_layers(64, 64, num_blocks[0], bit_width=bit_width, scale=scale)
        self.layer2 = self._make_layers(64, 128, num_blocks[1], stride=2, bit_width=bit_width, scale=scale)
        self.layer3 = self._make_layers(128, 256, num_blocks[2], stride=2, bit_width=bit_width, scale=scale)
        self.layer4 = self._make_layers(256, 512, num_blocks[3], stride=2, bit_width=bit_width, scale=scale)

        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            self._noise_factory.get_noise_operator(),
        )
        self.fc = nn.Sequential(
            #nn.Linear(512, num_classes),
            qnn.QuantLinear(in_features=512, out_features=num_classes, weight_bit_width=bit_width, bias=False, weight_quant=CommonIntWeightPerTensorQuant, ),
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
    
    def _make_layers(self, in_channels, out_channels, num_block, stride=1, bit_width=8, scale=None):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                #nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),  ## kernel 1 or 3?
                qnn.QuantConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, weight_bit_width=bit_width, bias=True, weight_quant=CommonIntWeightPerChannelQuant,),
                self._noise_factory.get_noise_operator(),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(
            BlockQAT(
                in_channels, out_channels, stride, downsample, noise_operator=self._noise_factory, bit_width=bit_width, scale=scale
            )
        )
        for _ in range(1, num_block):
            layers.append(
                BlockQAT(
                    out_channels,out_channels, noise_operator=self._noise_factory, bit_width=bit_width, scale=scale
                )
            )

        return nn.Sequential(*layers)

