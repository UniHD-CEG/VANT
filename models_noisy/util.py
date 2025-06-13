from torch.nn.modules.batchnorm import _NormBase
from noise_operator.operators import NoiseOperator
from noise_operator.config import NoNoiseConfig

from brevitas.quant import Uint8ActPerTensorFloat, Int8ActPerTensorFloat, Int8WeightPerTensorFloat

from brevitas.core.scaling import ScalingImplType

# quantizer
class CommonIntWeightPerTensorQuant(Int8WeightPerTensorFloat):
    """
    Common per-tensor weight quantizer with bit-width set to None so that it's forced to be
    specified by each layer.
    """
    scaling_min_val = 2e-16
    bit_width = None
class CommonIntWeightPerChannelQuant(CommonIntWeightPerTensorQuant):
    """
    Common per-channel weight quantizer with bit-width set to None so that it's forced to be
    specified by each layer.
    """
    scaling_per_output_channel = True
class ConstScaledActQuantUint(Uint8ActPerTensorFloat):
    scaling_impl_type = ScalingImplType.CONST
    min_val = 0.
    max_val = None # 1 is too small for 4-bit

class ConstScaledActQuantInt(Int8ActPerTensorFloat):
    scaling_impl_type = ScalingImplType.CONST
    min_val = None
    max_val = None # 1 is too small for 4-bit

class WeightClamper:
    """
    Class for clamping the weights of a given model.
    Inspired by: https://stackoverflow.com/a/70330290
    """
    def __init__(self, min_clamp=None, max_clamp=None):
        self._min = min_clamp
        self._max = max_clamp

    def __call__(self, module):
        # Only continue if something is to be done
        if (self._max is None) and (self._min is None):
            return
        # Only consider layer, which have weights
        if hasattr(module, 'weight'):
            # Skip Batchnorm layers
            if not issubclass(type(module), _NormBase):
                # Clamp weights
                w = module.weight.data
                w = w.clamp(self._min, self._max)
                module.weight.data = w


class ReConfigNoise:
    """
    Class for reseting the noise, used in incremental training 
    """
    def __init__(self, noise_config=NoNoiseConfig()):
        self.noise_config = noise_config

    def __call__(self, module):
        # Only continue if something is to be done
        #if self.noise_config.GaussStd == 0.0:
        #    return
        # Only consider layers which are NoiseOperator
        if isinstance(module, NoiseOperator):
            module.reset_noise(self.noise_config)
