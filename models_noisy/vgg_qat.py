### refer to: https://github.com/Xilinx/brevitas/blob/master/src/brevitas_examples/imagenet_classification/models/vgg.py
import torch
from torch import nn
from brevitas.nn import QuantConv2d
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantLinear
from brevitas.nn import QuantReLU
from brevitas.nn import TruncAvgPool2d

from noise_operator.config import NoNoiseConfig
from noise_operator.factory import NoiseOperatorFactory

#from models_noisy.common import *
from models_noisy.util import *

cfg = {
    'VGG11': [64, 'BN2d', 'ReLU', 'M', 128, 'BN2d', 'ReLU', 'M', 256, 'BN2d', 'ReLU', 256, 'BN2d', 'ReLU', 'M', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 'M', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 'M'],
    'VGG11-BN-noNoise': [64, 'BN2d-NN', 'ReLU', 'M', 128, 'BN2d-NN', 'ReLU', 'M', 256, 'BN2d-NN', 'ReLU', 256, 'BN2d-NN', 'ReLU', 'M', 512, 'BN2d-NN', 'ReLU', 512, 'BN2d-NN', 'ReLU', 'M', 512, 'BN2d-NN', 'ReLU', 512, 'BN2d-NN', 'ReLU', 'M'],
    'VGG11-noBN': [64, 'ReLU', 'M', 128, 'ReLU', 'M', 256, 'ReLU', 256, 'ReLU', 'M', 512, 'ReLU', 512, 'ReLU', 'M', 512, 'ReLU', 512, 'ReLU', 'M'],
    'VGG11_mnist': [64, 'BN2d', 'ReLU', 'M', 128, 'BN2d', 'ReLU', 'M', 256, 'BN2d', 'ReLU', 256, 'BN2d', 'ReLU', 'M', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 'M', 512, 'BN2d', 'ReLU', 512],
    'VGG11_mnist-BN-noNoise': [64, 'BN2d-NN', 'ReLU', 'M', 128, 'BN2d-NN', 'ReLU', 'M', 256, 'BN2d-NN', 'ReLU', 256, 'BN2d-NN', 'ReLU', 'M', 512, 'BN2d-NN', 'ReLU', 512, 'BN2d-NN', 'ReLU', 'M', 512, 'BN2d-NN', 'ReLU', 512],
    'VGG11_mnist-noBN': [64, 'ReLU', 'M', 128, 'ReLU', 'M', 256, 'ReLU', 256, 'ReLU', 'M', 512, 'ReLU', 512, 'ReLU', 'M', 512, 'ReLU', 512],
    'VGG13': [64, 'BN2d', 'ReLU', 64, 'BN2d', 'ReLU', 'M', 128, 'BN2d', 'ReLU', 128, 'BN2d', 'ReLU', 'M', 256, 'BN2d', 'ReLU', 256, 'BN2d', 'ReLU', 'M', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 'M', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 'M'],
    'VGG16': [64, 'BN2d', 'ReLU', 64, 'BN2d', 'ReLU', 'M', 128, 'BN2d', 'ReLU', 128, 'BN2d', 'ReLU', 'M', 256, 'BN2d', 'ReLU', 256, 'BN2d', 'ReLU', 256, 'BN2d', 'ReLU', 'M', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 'M', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 'M'],
    'VGG19': [64, 'BN2d', 'ReLU', 64, 'BN2d', 'ReLU', 'M', 128, 'BN2d', 'ReLU', 128, 'BN2d', 'ReLU', 'M', 256, 'BN2d', 'ReLU', 256, 'BN2d', 'ReLU', 256, 'BN2d', 'ReLU', 256, 'BN2d', 'ReLU', 'M', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 'M', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 512, 'BN2d', 'ReLU', 'M'],
# Potential ToDo: try out a QAT vs. non-QAT network experiment without noise injection after the ReLU, BN and MaxPool layers
}
min_max_dict = {
    4: (8, 15),
    8: (128, 256),
    16: (32768, 65536)
}

class VGG_QAT(nn.Module):
    def __init__(self,
                 conf_name='VGG11',
                 input_shape=(1, 28, 28),
                 default_noise_config=NoNoiseConfig(),
                 layer_wise_noise_config=dict(),
                 bit_width=8,
                 num_classes=10,
                 dropout=0.5,
                 scale=None,
                 ):
        super().__init__()
        # Makes some noise xD
        self._noise_factory = NoiseOperatorFactory(default_noise_config, layer_wise_config=layer_wise_noise_config)
        self.act_quant = {}
        if scale is not None:
            max_uint = min_max_dict[bit_width][1] * scale
            max_int = min_max_dict[bit_width][0] * scale
            self.act_quant["act_quant"] = ConstScaledActQuantUint
            self.act_quant["max_val"] = max_uint
            #self.relu = QuantReLU(bit_width=bit_width, return_quant_tensor=False, act_quant=ConstScaledActQuantUint, max_val = max_uint)
        self.features, output_channels = self._make_layers(cfg[conf_name], input_shape, bit_width)
        self.avgpool = TruncAvgPool2d(kernel_size=1, stride=1, bit_width=bit_width)

        # the input dimension of FC should be: output_channels *  input_shape[1]* 2**(-5)
        # because the size of the convolutional is 3, and there are 5 maxpolling layer
        shape_1 = round(input_shape[1]* 2**(-5))
        shape_2 = round(input_shape[2]* 2**(-5))
        in_channels_fc = output_channels * shape_1 * shape_2
        self.classifier = nn.Sequential(
            #QuantIdentity(bit_width=bit_width, act_quant=ConstScaledActQuantInt),
            QuantLinear(in_features=in_channels_fc, out_features=4096, weight_bit_width=bit_width, bias=True, weight_quant=CommonIntWeightPerChannelQuant),
            self._noise_factory.get_noise_operator(),
            QuantReLU(bit_width=bit_width, return_quant_tensor=False, **self.act_quant),
            self._noise_factory.get_noise_operator(),
            nn.Dropout(p=dropout),
            
            #QuantIdentity(bit_width=bit_width, act_quant=ConstScaledActQuantInt),
            QuantLinear(in_features=4096, out_features=4096, weight_bit_width=bit_width, bias=True, weight_quant=CommonIntWeightPerChannelQuant),
            self._noise_factory.get_noise_operator(),
            QuantReLU(bit_width=bit_width, return_quant_tensor=False, **self.act_quant),
            self._noise_factory.get_noise_operator(),
            nn.Dropout(p=dropout),
            
            QuantLinear(in_features=4096, out_features=num_classes, weight_bit_width=bit_width, bias=False, weight_quant=CommonIntWeightPerTensorQuant),
            #nn.Linear(4096, num_classes),
            self._noise_factory.get_noise_operator(),
            # ToDo: Output Quant
        )

        print(f"Created the following number of noise layers / operators: {self._noise_factory._layer_counter}")
        # Check for out of bounds layer noise configurations
        if self._noise_factory.check_for_unused_configs():
            raise ValueError(f"A noise setting for a layer not contained in the network was requested. "
                             f"This is likely due to an incorrect configuration. "
                             f"The built network has {self._noise_factory._layer_counter} noise layers, "
                             f"but layer wise configurations were requested for the following layer indices: "
                             f"{layer_wise_noise_config.keys()}")
        self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        #out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

    def _make_layers(self, config, input_shape, bit_width):
        layers = [
            self._noise_factory.get_noise_operator(),
        ]
        in_channels = input_shape[0]
        for x in config:
            if x == 'M':
                layers += [
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    self._noise_factory.get_noise_operator()
                ]
            elif type(x) == int:
                layers += [
                    #QuantIdentity(bit_width=bit_width, act_quant=ConstScaledActQuantInt),
                    QuantConv2d(in_channels=in_channels, out_channels=x, kernel_size=3, padding=1, 
                                    weight_bit_width=bit_width, bias=True, weight_quant=CommonIntWeightPerChannelQuant),
                    self._noise_factory.get_noise_operator(),
                ]
                in_channels = x
            elif x == 'BN2d':
                layers += [
                    nn.BatchNorm2d(in_channels),
                    self._noise_factory.get_noise_operator(),
                ]
            elif x == 'BN2d-NN':
                layers += [
                    nn.BatchNorm2d(in_channels),
                ]
            elif x == 'ReLU':
                layers += [
                    QuantReLU(bit_width=bit_width, return_quant_tensor=False, **self.act_quant),
                    self._noise_factory.get_noise_operator(),
                ]
            else:
                raise NotImplementedError
        
        return nn.Sequential(*layers), in_channels
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
