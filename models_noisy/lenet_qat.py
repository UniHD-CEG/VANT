from torch import nn
from brevitas import nn as qnn

from noise_operator.config import NoNoiseConfig
from noise_operator.factory import NoiseOperatorFactory
from noise_operator.operators import NoiseOperator

from models_noisy.util import *
#from models_noisy.common import *



cfg = {
    'LeNet5': ['C6@5', 'ReLU', 'M2', 'C16@5', 'ReLU', 'M2', 'Flatten', 'FC120', 'ReLU', 'FC84', 'ReLU', ],
    'LeNet5-BN': ['C6@5', 'BN2d', 'ReLU', 'M2', 'C16@5', 'BN2d', 'ReLU', 'M2', 'Flatten', 'FC120', 'BN1d',
                  'ReLU', 'FC84', 'BN1d', 'ReLU', ],
    'LeNet5-BN-noNoise': ['C6@5', 'BN2d-NN', 'ReLU', 'M2', 'C16@5', 'BN2d-NN', 'ReLU', 'M2', 'Flatten', 'FC120',
                          'BN1d-NN',
                          'ReLU', 'FC84', 'BN1d-NN', 'ReLU', ],
# Potential ToDo: try out a QAT vs. non-QAT network experiment without noise injection after the ReLU, BN and MaxPool layers
}
min_max_dict = {
    4: (8, 15),
    8: (128, 256),
    16: (32768, 65536)
}

class LeNetQAT(nn.Module):
    """
    LeNet network, implemnted as described in the original paper on page 7: https://www.researchgate.net/profile/Yann-Lecun/publication/2985446_Gradient-Based_Learning_Applied_to_Document_Recognition/links/0deec519dfa1983fc2000000/Gradient-Based-Learning-Applied-to-Document-Recognition.pdf?origin=publication_detail
    Also inspired by: https://github.com/ChawDoe/LeNet5-MNIST-PyTorch/blob/master/model.py
    """
    def __init__(self,
                 conf_name='LeNet5',
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
        self.features, output_channels = self._make_layers(cfg[conf_name], input_shape, bit_width, scale)
        self.features.extend([
            qnn.QuantLinear(output_channels, num_classes, weight_bit_width=bit_width, bias=False, weight_quant=CommonIntWeightPerTensorQuant),
            #nn.Linear(output_channels, num_classes),
            self._noise_factory.get_noise_operator(),
            # ToDo: Insert QuantIdentity as output quant
            ])

        num_noise_layers = self._noise_factory._layer_counter
        print(f"Created the following number of noise layers / operators: {num_noise_layers}")
        # Check for out of bounds layer noise configurations
        if self._noise_factory.check_for_unused_configs():
            raise ValueError(f"A noise setting for a layer not contained in the network was requested. "
                             f"This is likely due to an incorrect configuration. "
                             f"The built network has {num_noise_layers} noise layers, "
                             f"but layer wise configurations were requested for the following layer indices: "
                             f"{layer_wise_noise_config.keys()}")


    def forward(self, x):
        for mod in self.features:
            x = mod(x)
        return x

    def _make_layers(self, config, input_shape, bit_width, scale):
        if scale is not None:
            max_uint = min_max_dict[bit_width][1] * scale
            max_int = min_max_dict[bit_width][0] * scale
        layers = [
            self._noise_factory.get_noise_operator(),
            #qnn.QuantIdentity(bit_width=bit_width, act_quant=ConstScaledActQuantInt),
        ]
        in_channels = input_shape[0]
        # manual calculating, two conv with kernel 5 and two maxpool with kernel 2
        # the QuantLinear do not accept lazy mode
        shape_1 = round((round((input_shape[1]-4)/2.0) - 4)/2.0)
        shape_2 = round((round((input_shape[2]-4)/2.0) - 4)/2.0)
        in_channels_fc = shape_1 * shape_2
        for x in config:
            if x.startswith('M'):
                kernel_size = int(x.split('M')[-1])
                layers += [
                    nn.MaxPool2d(kernel_size),
                    self._noise_factory.get_noise_operator(),
                ]
            elif x == 'ReLU':
                if scale is not None:
                    layers += [qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=False, 
                                    act_quant=ConstScaledActQuantUint, max_val=max_uint),
                    ]
                else:
                    layers += [qnn.QuantReLU(bit_width=bit_width, return_quant_tensor=False)]
                layers += [
                    self._noise_factory.get_noise_operator(),
                ]
            elif x.startswith('FC'):
                # no LazyLinear
                num_ch = int(x.split('FC')[-1])
                if in_channels_fc != 0:
                    layers += [
                        #qnn.QuantIdentity(bit_width=bit_width,  act_quant=ConstScaledActQuantInt),
                        qnn.QuantLinear(in_features=in_channels_fc * in_channels, out_features=num_ch, weight_bit_width=bit_width, bias=True, weight_quant=CommonIntWeightPerTensorQuant, ),
                        self._noise_factory.get_noise_operator(),
                    ]
                    in_channels_fc = 0
                else:
                    layers += [
                        #qnn.QuantIdentity(bit_width=bit_width, act_quant=ConstScaledActQuantInt),
                        qnn.QuantLinear(in_features=in_channels, out_features=num_ch, weight_bit_width=bit_width, bias=True, weight_quant=CommonIntWeightPerTensorQuant, ),
                        self._noise_factory.get_noise_operator(),
                    ]
                in_channels = num_ch
            elif x.startswith('C'):
                info_list = x.split('C')[-1].split('@')
                num_ch = int(info_list[0])
                kernel_size = int(info_list[1])
                layers += [
                    #qnn.QuantIdentity(bit_width=bit_width, act_quant=ConstScaledActQuantInt),
                    qnn.QuantConv2d(in_channels=in_channels, out_channels=num_ch, kernel_size=kernel_size, weight_bit_width=bit_width, bias=True, weight_quant=CommonIntWeightPerChannelQuant,),
                    self._noise_factory.get_noise_operator(),
                ]
                in_channels = num_ch
            elif x == 'BN2d':
                layers += [
                    nn.BatchNorm2d(in_channels),
                    self._noise_factory.get_noise_operator(),
                ]
            elif x == 'BN1d':
                layers += [
                    nn.BatchNorm1d(in_channels),
                    self._noise_factory.get_noise_operator(),
                ]
            elif x == 'BN2d-NN':
                layers += [
                    nn.BatchNorm2d(in_channels),
                ]
            elif x == 'BN1d-NN':
                layers += [
                    nn.BatchNorm1d(in_channels),
                ]
            elif x == 'Flatten':
                layers += [
                    nn.Flatten(),
                ]
            else:
                raise NotImplementedError
        return nn.ModuleList(layers), in_channels
