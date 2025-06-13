import copy
import filecmp
import hashlib
import json
import os
import os.path
import shutil
import subprocess
import time
import uuid
from io import StringIO
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import seml
import torch
import torch.nn as nn
from torch import autograd
from torchvision import transforms
from sacred import Experiment
#from torchattacks import PGD, GN, DeepFool, AutoAttack


from data.CIFAR10 import CIFAR10
from data.FashionMNIST import FashionMNIST
from data.GSC2 import GSC2_PyTorch, GSC2_TF
from data.MNIST import MNIST
from data.SVHN import SVHN
from data.TinyImageNet import TinyImageNet
from data.CINIC10 import CINIC10
from models_noisy.cnn_HE import CNN_HE
from models_noisy.lenet import LeNet
from models_noisy.mlp import MLP
from models_noisy.lenet_qat import LeNetQAT
from models_noisy.vgg_qat import VGG_QAT
from models_noisy.util import WeightClamper
from models_noisy.util import ReConfigNoise
from models_noisy.vgg import VGG
from models_noisy.resnet_new import ResNet
from models_noisy.resnet_qat import ResNetQAT
from models_noisy.densenet import DenseNet
from noise_operator import config as cfg
from util import cluster
from util.console_logging import print_status
from util.attacks import *
from util.awp import AdvWeightPerturb
from util.sam import SAM

sacred_exp = Experiment(save_git_info=False)
seml.setup_logger(sacred_exp)


@sacred_exp.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@sacred_exp.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        sacred_exp.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))
        # sacred_exp.observers.append(seml.create_neptune_observer('uhdcsg/' + db_collection, api_token=None))


class ExperimentWrapper:
    """
    A simple wrapper around a sacred experiment, making use of sacred's captured functions with prefixes.
    This allows a modular design of the configuration, where certain sub-dictionaries (e.g., "data") are parsed by
    specific method. This avoids having one large "main" function which takes all parameters as input.
    """

    def __init__(self,
                 sacred_exp,
                 init_all=True,
                 nfs_artifact_root=None,
                 device=None,
                 ):
        # Setup internal variables
        self._sacred_exp = sacred_exp
        self._seml_return_data = dict()
        self._seml_return_data['artifacts'] = dict()
        if nfs_artifact_root is None:
            self._nfs_artifact_root = cluster.get_artifact_root()
        elif nfs_artifact_root:
            self._nfs_artifact_root = nfs_artifact_root
        try:
            self.num_avail_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
            self.num_avail_cpus = min(3, self.num_avail_cpus)
        except KeyError:
            self.num_avail_cpus = 1
        
        self.hostname = os.environ.get("SLURM_NODELIST", "Unknown")

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        # Generate the artifact UUID, we trust that the config-hash and UUID4 together are sufficiently unique here
        config_hash = self.get_config_hash()
        self._artifact_uuid = f'{config_hash}_{str(uuid.uuid4())}'
        # Do main initialization
        if init_all:
            self.init_all()

    def get_seml_return_data(self):
        return self._seml_return_data

    def log_scalar(self, metric_name, value, log_to_sacred_observers=False):
        """
        Wrapper function, which logs to sacred and to the seml return argument at the same time.
        """
        # Log to sacred observers
        if log_to_sacred_observers:
            self._sacred_exp.log_scalar(metric_name, value)
        # Log to seml
        log_time = time.time()
        if metric_name not in self._seml_return_data.keys():
            self._seml_return_data[metric_name] = list()
            self._seml_return_data[metric_name + "_time"] = list()
        self._seml_return_data[metric_name].append(value)
        self._seml_return_data[metric_name + "_time"].append(log_time)

    @sacred_exp.capture
    def get_config_hash(self, general, data, model, optimizer, noise_settings, db_collection):
        """Calculates the sha1 hash value of the current experiment configuration
        Inspired from: https://github.com/TUM-DAML/seml/blob/7d9352e51c9a83b77aa30617e8926863f0f48797/seml/utils.py#L191"""
        config_hash = hashlib.sha1()
        config_dict = {'general': general, 'data': data, 'model': model, 'optimizer': optimizer,
                       'noise_settings': noise_settings, 'db_collection': db_collection}
        config_hash.update(json.dumps(config_dict, sort_keys=True).encode("utf-8"))
        return config_hash.hexdigest()

    def add_artifact(self, filename: Union[str, Path],
                     name: Optional[str] = None,
                     metadata: Optional[dict] = None,
                     content_type: Optional[str] = None,
                     ) -> None:
        """
        Copies the artifact to the network file system and stores its path in the seml result data.
        """
        # Create a new artifact folder
        artifact_folder = self._nfs_artifact_root / self._artifact_uuid
        artifact_folder.mkdir()

        # Copy over the file
        src_file = Path(filename)
        dst_file = artifact_folder / src_file.name
        shutil.copy2(src_file, dst_file)
        print(f"Copied artifact {src_file} to {dst_file}")
        # Check that the file was copied correctly and overwrite otherwise
        for i in range(5):
            time.sleep(1.)
            filecmp.clear_cache()
            if not filecmp.cmp(src_file, dst_file, shallow=False):
                print(f"Artifact copy mismatches the original, retrying ({i}-th try).")
                shutil.copy2(src_file, dst_file)
            else:
                break
        else:  # No break executed
            print("Copy operation of the artifact failed after too many retries.")
        # Save the artifact name in the seml data
        if name is None:
            name = src_file.name
        self._seml_return_data['artifacts'][name] = str(dst_file)

    # With the prefix option we can "filter" the configuration for the sub-dictionary under "data".
    @sacred_exp.capture()
    def init_dataset(self, data):
        """
        Perform dataset loading, preprocessing etc.
        Since we set prefix="data", this method only gets passed the respective sub-dictionary, enabling a modular
        experiment design.
        """
        # Find the dataset
        dataset = data['dataset']
        if dataset == "MNIST":
            self.data = MNIST
        elif dataset == "CIFAR10":
            self.data = CIFAR10
        elif dataset == "GSC2_PyTorch":
            self.data = GSC2_PyTorch
        elif dataset == "GSC2_TF":
            self.data = GSC2_TF
        elif dataset == "SVHN":
            self.data = SVHN
        elif dataset == "FashionMNIST":
            self.data = FashionMNIST
        elif dataset == "TinyImageNet":
            self.data = TinyImageNet
        elif dataset == "CINIC10":
            self.data = CINIC10
        else:
            raise ValueError(f"Dataset with name {dataset} is not supported.")
        # Get the batch_size
        
        if 'batch_size' in data:
            batch_size = data['batch_size']
        else:
            batch_size = 128
        
        #batch_size = 256
        # Init the dataset
        #if dataset == "TinyImageNet" and self.hostname.find("rivulet") >= 0:
        #    self.data = self.data(num_workers=self.num_avail_cpus,
        #                      data_root=str(Path('/local/') / Path('datasets/')),
        #                      batch_size=batch_size,
        #                      )
        
        self.data = self.data(num_workers=self.num_avail_cpus,
                            data_root=str(cluster.get_artifact_root() / Path('datasets/')),
                            batch_size=batch_size,
                            )
        # Download the data
        self.data.prepare_data()
        # Setup the torch datasets
        self.data.setup()

        # Init train and val dataloaders
        self.train_loader = self.data.train_dataloader()
        self.val_loader = self.data.val_dataloader()
        if self.data.has_test_dataset:
            self.test_loader = self.data.test_dataloader()

    @staticmethod
    def assemble_layer_noise_config(experiment_noise_settings):
        '''
        Create the layer wise configuration for the noise factory from the experiment configuration.
        '''
        layer_wise_noise_config = dict()

        # Check for a layer wise setting, this addresses only one layer
        # Check that we have a sub dictionary available to us
        # Otherwise this might just be none and we can continue.
        try:
            layer_wise_setting = experiment_noise_settings['layer_wise']
        except KeyError:
            layer_wise_setting = None
        if isinstance(layer_wise_setting, dict):
            # For now only configuring one single layer is supported
            ex_layer_settings = experiment_noise_settings['layer_wise']
            single_layer_config = cfg.resolve_config_from_name(
                ex_layer_settings['noise_type'],
                **ex_layer_settings
            )
            index = int(ex_layer_settings['layer_index'])
            layer_wise_noise_config[index] = single_layer_config
            return layer_wise_noise_config

        # Check for a layer mapped setting, this addresses multiple layers
        try:
            layer_mapped_setting = experiment_noise_settings['layer_mapped']
        except KeyError:
            layer_mapped_setting = None
        if isinstance(layer_mapped_setting, dict):
            mapped_settings = copy.deepcopy(layer_mapped_setting)
            std_map = mapped_settings['std_map']
            # Do mapping rescaling
            std_multiplication_factor = mapped_settings['std_multiplication_factor']
            if mapped_settings['re_normalize_mapping']:
                std_sum = sum(std_map.values())
                std_multiplication_factor /= std_sum
            for key in std_map.keys():
                std_map[key] *= std_multiplication_factor
            # Create mapping
            for index in std_map.keys():
                kwarg_settings = copy.deepcopy(mapped_settings['noise_op_kwargs'])
                kwarg_settings[mapped_settings['std_key_name']] = std_map[index]
                single_layer_config = cfg.resolve_config_from_name(
                    mapped_settings['noise_type'],
                    **kwarg_settings
                )
                index = int(index)
                layer_wise_noise_config[index] = single_layer_config
            # Done
            return layer_wise_noise_config

        return layer_wise_noise_config

    def create_model(self, model, default_noise_config, layer_wise_noise_config):
        # Setup the nn
        model_class = model['model_class']
        if model_class == "MLP":
            # Here we can pass the "model_params" dict to the constructor directly, which can be very useful in
            # practice, since we don't have to do any model-specific processing of the config dictionary.
            internal_model = MLP(**model['MLP'],
                             num_classes=self.data.num_classes,
                             input_shape=self.data.dims,
                             default_noise_config=default_noise_config,
                             layer_wise_noise_config=layer_wise_noise_config)
        elif model_class == 'VGG':
            internal_model = VGG(**model['VGG'],
                             num_classes=self.data.num_classes,
                             input_shape=self.data.dims,
                             default_noise_config=default_noise_config,
                             layer_wise_noise_config=layer_wise_noise_config)
        elif model_class == 'LeNet':
            internal_model = LeNet(**model['LeNet'],
                               num_classes=self.data.num_classes,
                               input_shape=self.data.dims,
                               default_noise_config=default_noise_config,
                               layer_wise_noise_config=layer_wise_noise_config)
        elif model_class == "CNN_HE":
            internal_model = CNN_HE(**model['CNN_HE'],
                                num_classes=self.data.num_classes,
                                input_shape=self.data.dims,
                                default_noise_config=default_noise_config,
                                layer_wise_noise_config=layer_wise_noise_config)
        elif model_class == 'LeNet_QAT':
            internal_model = LeNetQAT(**model['LeNet_QAT'],
                               num_classes=self.data.num_classes,
                               input_shape=self.data.dims,
                               default_noise_config=default_noise_config,
                               layer_wise_noise_config=layer_wise_noise_config)
        elif model_class == 'VGG_QAT':
            internal_model = VGG_QAT(**model['VGG_QAT'],
                               num_classes=self.data.num_classes,
                               input_shape=self.data.dims,
                               default_noise_config=default_noise_config,
                               layer_wise_noise_config=layer_wise_noise_config)
        elif model_class == 'ResNet':
            internal_model = ResNet(**model['ResNet'],
                               num_classes=self.data.num_classes,
                               input_shape=self.data.dims,
                               default_noise_config=default_noise_config,
                               layer_wise_noise_config=layer_wise_noise_config)
        elif model_class == 'ResNet_QAT':
            internal_model = ResNetQAT(**model['ResNet_QAT'],
                               num_classes=self.data.num_classes,
                               input_shape=self.data.dims,
                               default_noise_config=default_noise_config,
                               layer_wise_noise_config=layer_wise_noise_config)
        elif model_class == 'DenseNet':
            internal_model = DenseNet(**model['DenseNet'],
                               num_classes=self.data.num_classes,
                               input_shape=self.data.dims,
                               default_noise_config=default_noise_config,
                               layer_wise_noise_config=layer_wise_noise_config)
        else:
            raise ValueError(f"Model with name {model_class} is not supported.")
        # Move the model to the correct device and set the datatype to float32
        internal_model.to(self.device, dtype=torch.float32)
        # Do a dry run to initialize the lazy-init layers
        init_tensor = torch.randn(2, *self.data.dims, dtype=torch.float32, device=self.device)
        internal_model(init_tensor)
        return internal_model

    @sacred_exp.capture()
    def init_model(self, general, model, noise_settings, data, _log, kl_div_metric=None):
        # Setup the noise, quantized, pruned, whatever default model
        # Get default and layer wise noise settings
        default_noise_config = cfg.resolve_config_from_name(
            noise_settings['default']['noise_type'],
            **noise_settings['default']
        )
        layer_wise_noise_config = self.assemble_layer_noise_config(noise_settings)

        self.model = self.create_model(model, default_noise_config, layer_wise_noise_config)
        # load from pre-trained model
        # use another parameter, enable_pretrain, and then model_dir
        if "enable_pretrain" in model and model["enable_pretrain"]:
            pre_train_model = torch.load(model["pretrain_dir"])
            self.model.load_state_dict(pre_train_model.state_dict(), strict=False)
        if "distill" in model and model["distill"]["is_enabled"]:
            self.teacher = torch.load(model["distill"]["teacher"])
        if "enable_awp" in general and general["enable_awp"]:
            self.proxy_model = self.create_model(model, default_noise_config, layer_wise_noise_config)
        if "param_sigma" in noise_settings and noise_settings["param_sigma"]:
            self.pure_model = torch.load(noise_settings["param_sigma"]["clean_model"])
            self.model.load_state_dict(self.pure_model.state_dict(), strict=False)
            # set up noise in pure model
            self.pure_model.apply(ReConfigNoise(default_noise_config)).to(self.device)
            # set up noise in model
            parametric_conf = {
                "GaussMean": 0.0,
                "GaussStd": noise_settings["param_sigma"]["init_Std"],
                "enable_in_training": 1,
            }
            parametric_noise_config = cfg.resolve_config_from_name(noise_settings["default"]["noise_type"], **parametric_conf)
            self.model.apply(ReConfigNoise(parametric_noise_config)).to(self.device)

        # Print check
        _log.info(f"Created model of the following configuration: {self.model}")

        # Check for more models, which need setting up
        # KL_div models
        if kl_div_metric is not None:
            if kl_div_metric['compute_against_pre_trained_no_noise']:
                self.compute_kl_against_no_noise = True
                # Create the reference model from a non-noisy checkpoint
                # First get the checkpoint from the NFS share
                pre_trained_db_entry = get_no_noise_db_entry_equivalent_to_current_exp()
                found_pre_trained_model = False
                if pre_trained_db_entry is None:
                    _log.warning('No equivalent pre-trained model was found, '
                                 'evaluating this experiment may take longer than what would otherwise be required.')
                else:
                    found_pre_trained_model, pre_trained_checkpoint = load_checkpoint_from_exp(pre_trained_db_entry, device=self.device)
                if not found_pre_trained_model:
                    raise RuntimeError("Could not find a pretrained no-noise experiment, which is required to compute the KL-divergence")
                # Then create the model, w/o noise
                default_noise_config = cfg.resolve_config_from_name(
                    'NoNoise',
                )
                layer_wise_noise_config = self.assemble_layer_noise_config({'layer_wise': None})
                self.kl_no_noise_model = self.create_model(model, default_noise_config, layer_wise_noise_config)
                # Load the weights
                self.kl_no_noise_model.load_state_dict(pre_trained_checkpoint['state_dict'])
            else:
                self.compute_kl_against_no_noise = False
        else:
            self.compute_kl_against_no_noise = False


        # Setup the criterion
        crit_name = model['criterion']
        if crit_name == "CrossEntropyLoss":
            loss_weight = None
            # Re-weighting for GSC2_TF
            if 'weight_criterion' in model:
                if model['weight_criterion']:
                    if data['dataset'] == "GSC2_TF":
                        # Compute weights to balance the training dataset
                        bins, edges = np.histogram(self.data._ds_train._label_array, 12)
                        class_density = bins / bins.sum()
                        inverse_class_density = 1 - class_density
                        # Further suppress the "unknown" label, since it is severely overrepresented in training data
                        # The exact value is taken from here: https://github.com/mlcommons/tiny_results_v0.7/blob/691f8b26aa9dffa09b1761645d4a35ad35a4f095/open/hls4ml-finn/code/kws/KWS-W3A3/training/const_QMLP.yaml#L32
                        label_suppression_unknown = 3.6
                        inverse_class_density[-1] /= label_suppression_unknown
                        inverse_class_density /= inverse_class_density.sum()
                        inverse_class_density = inverse_class_density.astype(np.float32)
                        loss_weight = torch.from_numpy(inverse_class_density).to(self.device)
                    else:
                        ValueError("Weighting is currently not implemented for other datasets than GSC2_TF.")
            self.criterion = nn.CrossEntropyLoss(weight=loss_weight)
        else:
            raise ValueError(f"Criterion with name {crit_name} is not supported.")

    @sacred_exp.capture()
    def init_optimizer(self, general, optimizer):
        # Set the optimizer
        optim_name = optimizer['optimizer_type']
        weight_decay = optimizer.get("weight_decay", 0)
        if optim_name == "Adam":
            if "enable_sam" in general and general["enable_sam"]:
                is_adaptive = general["is_adaptive"]
                rho = general["sam_rho"]
                self.optimizer = SAM(self.model.parameters(),  torch.optim.Adam, rho=rho,  adaptive=is_adaptive, lr=optimizer['lr'], weight_decay=weight_decay)
            else:
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=optimizer['lr'], weight_decay=weight_decay)
                if "enable_awp" in general and general["enable_awp"]:
                    self.proxy_optim = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=weight_decay)
                    gamma = general["awp_gamma"]
                    self.awp_adversary = AdvWeightPerturb(model=self.model, proxy=self.proxy_model, proxy_optim=self.proxy_optim, gamma=gamma)
        else:
            raise ValueError(f"Optimizer with name {optim_name} is not supported.")

        # Set the scheduler
        sched_name = optimizer['lr_scheduler']
        if sched_name == "CosineAnnealingLR":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=general['num_epochs'])
        else:
            raise ValueError(f"Scheduler with name {sched_name} is not supported.")

        # Figure out the clamping and put in none if it wasn't specified
        try:
            weight_clamping_params = [general['weight_clamping']['min'], general['weight_clamping']['max']]
        except KeyError:
            weight_clamping_params = [None, None]
        self._weightClamper = WeightClamper(*weight_clamping_params)

    def init_all(self):
        """
        Sequentially run the sub-initializers of the experiment.
        """
        self.init_dataset()
        self.init_model()
        self.init_optimizer()
    
    @sacred_exp.capture()
    def reset_noiseop_per_batch(self, noise_settings, _log, batch_size=128, enable_log=False):
        """ variance of variance
            std for each image is sampled from distribution N(GaussStd_in_training, Std_of_GaussStd_in_training)
        """
        mean_std = noise_settings["default"]["GaussStd_in_training"]
        std_of_std = noise_settings["default"]["Std_of_GaussStd_in_training"]
        if "is_mean_of_std" in noise_settings and noise_settings["is_mean_of_std"]:
            mean_std = mean_std * noise_settings["default"]["Mean_of_GaussStd_in_training"]
        std_per_batch = list(mean_std + std_of_std * np.abs(np.random.randn(batch_size)))
        noise_info = noise_settings["default"]
        noise_new = copy.deepcopy(noise_info)
        noise_new["GaussStd"] = std_per_batch
        noise_config = cfg.resolve_config_from_name(noise_new["noise_type"], **noise_new)
        noise_case = ReConfigNoise(noise_config)
        self.model.apply(noise_case).to(self.device)
        #_log.info(f"info of std per batch, {len(std_per_batch), std_per_batch}")
        #_log.info(f"model after reset noiseoperator {self.model}")
        if enable_log:
            self.log_scalar("train_std_per_batch",  std_per_batch)

    @sacred_exp.capture()
    def training_step(self, optimizer, noise_settings, cur_train_std = None, ):
        # Training step
        self.model.train()
        summed_loss = 0.
        correct = 0
        total = 0
        if "GaussStd_in_training" in noise_settings["default"]:
            train_std = noise_settings["default"]["GaussStd_in_training"]
        else:
            train_std = noise_settings["default"].get("GaussStd", 0)
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            # if the "enable_std_per_img" is on, we only do that after the incremeantal training part. 
            if cur_train_std is not None and cur_train_std >= train_std:
                # apply individual std to each img
                if noise_settings.get("enable_std_per_img", False) and "Std_of_GaussStd_in_training" in noise_settings["default"]:
                    self.reset_noiseop_per_batch(batch_size=inputs.shape[0], enable_log=batch_idx<2)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            # add gradient penalty to loss
            if "gradient_reg" in optimizer:
                lambda_reg = optimizer["gradient_reg"]
                grad_outputs = torch.ones(loss.shape)
                gradients = autograd.grad(outputs=loss, inputs=self.model.parameters(), 
                                    create_graph=True)
                grad_l2_norm = 0
                for temp_grad in gradients:
                    grad_l2_norm += torch.norm(temp_grad, p=2)
                total_loss = loss + lambda_reg * grad_l2_norm
            else:
                total_loss = loss
            total_loss.backward()
            self.optimizer.step()

            # Apply weight clamping after optimization step
            self.model.apply(self._weightClamper)

            summed_loss += total_loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Logging
        accuracy = 100. * correct / total
        self.log_scalar("training.loss", summed_loss)
        self.log_scalar("training.accuracy", accuracy)

        return summed_loss, accuracy

    @sacred_exp.capture()
    def validation_step(self, general, run_on_test_dataset_instead=False):
        # Validation step
        topk = general.get("topk", 1)
        self.model.eval()
        summed_loss = 0.
        correct = 0
        total = 0
        distorted_probabilities = []
        reference_probabilities = []

        with torch.no_grad():
            curr_ds = self.val_loader
            if run_on_test_dataset_instead:
                curr_ds = self.test_loader
            for batch_idx, (inputs, targets) in enumerate(curr_ds):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                summed_loss += loss.item()
                total += targets.size(0)
                if topk == 1:
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(targets).sum().item()
                else:
                    _, predicted = outputs.topk(topk, dim=1)
                    correct += predicted.eq(targets.view(-1, 1)).any(dim=1).sum().item()
                if self.compute_kl_against_no_noise:
                    distorted_probabilities.append(torch.nn.functional.log_softmax(outputs, dim=1).detach())
                    self.kl_no_noise_model.eval()
                    ref_out = self.kl_no_noise_model(inputs)
                    reference_probabilities.append(torch.nn.functional.log_softmax(ref_out, dim=1).detach())
            if self.compute_kl_against_no_noise:
                distorted_probabilities = torch.cat(distorted_probabilities, dim=0)
                reference_probabilities = torch.cat(reference_probabilities, dim=0)
                kl_div = torch.nn.functional.kl_div(distorted_probabilities, reference_probabilities,
                                                    reduction='batchmean', log_target=True).cpu().item()
                log_name = 'test.kl_div' if run_on_test_dataset_instead else 'validation.kl_div'
                self.log_scalar(log_name, kl_div)

        # Logging
        accuracy = 100. * correct / total
        if run_on_test_dataset_instead:
            self.log_scalar("test.loss", summed_loss)
            self.log_scalar("test.accuracy", accuracy)
        else:
            self.log_scalar("validation.loss", summed_loss)
            self.log_scalar("validation.accuracy", accuracy)

        return summed_loss, accuracy
    
    @sacred_exp.capture()
    def training_step_adv(self, general):
        self.model.train()
        summed_loss = 0.
        correct = 0
        total = 0
        # Training step using the attack
        adv_type = general["adv_type"]
        """
        if adv_type == "FGSM":
            attack = FGSM(self.model, eps=4/255) # 8/255 as default
        elif adv_type == "PGD":
            # eps = 8/255, alpha = 1/255, and steps = 10 as default
            attack = PGD(self.model, eps=4/255, alpha=1/255, steps=3, random_start=True)
        """
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            if adv_type == "GN":
                trans = transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 0.75))
                inputs_adv = trans(inputs)
            elif adv_type == "PGD":
                inputs_adv, _, _, _ = PGD(self.model, inputs, targets)
            else:
                raise ValueError(f"The given adv_type is not expected.")
            #inputs_adv = attack(inputs, targets)
            self.optimizer.zero_grad()
            outputs = self.model(inputs_adv)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            # Apply weight clamping after optimization step
            self.model.apply(self._weightClamper)

            summed_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Logging
        accuracy = 100. * correct / total
        self.log_scalar("training.loss", summed_loss)
        self.log_scalar("training.accuracy", accuracy)

        return summed_loss, accuracy

    @sacred_exp.capture()
    def training_step_awp(self, general):
        self.model.train()
        summed_loss = 0.
        correct = 0
        total = 0
        # Training step using the attack
        adv_type = general.get("adv_type", None)
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            if adv_type == "PGD":
                inputs_adv, _, _, _ = PGD(self.model, inputs, targets)
                #attack = PGD(self.model, eps=8./255, alpha=2./255, steps=10)
                #inputs_adv = attack(inputs, targets)
            else:
                # none
                inputs_adv = inputs
            # awp 
            awp = self.awp_adversary.calc_awp(inputs_adv=inputs_adv,
                                             targets=targets)
            self.awp_adversary.perturb(awp) # model is perturbed

            self.optimizer.zero_grad()
            outputs = self.model(inputs_adv)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            self.awp_adversary.restore(awp)

            # Apply weight clamping after optimization step
            self.model.apply(self._weightClamper)

            summed_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Logging
        accuracy = 100. * correct / total
        #self.log_scalar("adv_type", adv_type)
        #self.log_scalar("gamma", general["awp_gamma"])
        self.log_scalar("training.loss", summed_loss)
        self.log_scalar("training.accuracy", accuracy)

        return summed_loss, accuracy

    def training_step_distill(self, model):
        T = model["distill"]["T"]
        b = model["distill"]["b"]
        self.teacher.eval()
        self.model.train()
        summed_loss = 0.
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            student_logits = self.model(inputs)
            #forward pass with the teacher model, do not save gradients here
            with torch.no_grad():
                teacher_logits = self.teacher(inputs)
            soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
            #soft_prob = nn.functional.softmax(student_logits / T, dim=-1)
            # Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
            #soft_loss = T**2 * torch.sum(soft_targets * (soft_targets.log() - soft_prob.log()))/student_logits.size()[0]
            soft_loss = T**2 * self.criterion(student_logits/T, soft_targets)
            
            label_loss = self.criterion(student_logits, targets)
            # Weighted sum of the two losses
            loss = (1-b) * label_loss + b * soft_loss
            loss.backward()
            self.optimizer.step()

            # Apply weight clamping after optimization step
            self.model.apply(self._weightClamper)

            summed_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Logging
        accuracy = 100. * correct / total
        self.log_scalar("training.loss", summed_loss)
        self.log_scalar("training.accuracy", accuracy)

        return summed_loss, accuracy
    
    def training_step_sam(self):
        # Training step
        self.model.train()
        summed_loss = 0.
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            #self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            #self.optimizer.step()
            self.optimizer.first_step(zero_grad=True)
            self.criterion(self.model(inputs), targets).backward()
            self.optimizer.second_step(zero_grad=True)

            # Apply weight clamping after optimization step
            self.model.apply(self._weightClamper)

            summed_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Logging
        accuracy = 100. * correct / total
        self.log_scalar("training.loss", summed_loss)
        self.log_scalar("training.accuracy", accuracy)

        return summed_loss, accuracy
    
    @sacred_exp.capture() 
    def training_step_noisy_weight(self, noise_settings):
        # Training step
        self.model.train()
        summed_loss = 0.
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            # step 0: get noise information, GaussAdd as default
            mean = 0
            std = 0
            noise_default_type = noise_settings["default"]["noise_type"]
            if noise_default_type == "GaussAdd":
                # global noise
                mean = noise_settings["default"]["GaussMean"]
                std = noise_settings["default"]["GaussStd"]
            # step 1: inject noise on weight
            src_params = []
            for p_name, p in self.model.named_parameters():
                if p_name.find('weight'):
                    src_params.append(p.clone())
                    p.data = p.data + torch.normal(mean, std, p.size(), device=self.device)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            #step 2: reset the weights and update
            for p_name, p in self.model.named_parameters():
                if p_name.find('weight'):
                    p.data = src_params.pop(0)
            self.optimizer.step()
            # Apply weight clamping after optimization step
            self.model.apply(self._weightClamper)

            summed_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        # Logging
        accuracy = 100. * correct / total
        self.log_scalar("training.loss", summed_loss)
        self.log_scalar("training.accuracy", accuracy)

        return summed_loss, accuracy
        
    @sacred_exp.capture()
    def training_param_sigmas(self, noise_settings):
        param_sigma = noise_settings["param_sigma"]
        optimizer = torch.optim.Adam(self.model.parameters(), lr=param_sigma["lr"], weight_decay=param_sigma["weight_decay"])
        num_epochs = param_sigma["num_epochs"]
        T = param_sigma["T"]
        b = param_sigma["b"]
        # set std learnable explicitly
        for p in self.model.named_parameters():
            #print(p[0], p[1].requires_grad)
            if p[0].find('weight')>=0 or p[0].find('bias')>=0 :
                p[1].requires_grad = False
            else:
                p[1].requires_grad = True
        for epoch in range(num_epochs):
            self.pure_model.eval()
            self.model.train()
            summed_loss = 0.
            correct = 0
            teacher_correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                student_logits = self.model(inputs)
                #forward pass with the teacher model, do not save gradients here
                with torch.no_grad():
                    teacher_logits = self.pure_model(inputs)
                soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
                #soft_prob = nn.functional.softmax(student_logits / T, dim=-1)
                # Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
                #soft_loss = T**2 * torch.sum(soft_targets * (soft_targets.log() - soft_prob.log()))/student_logits.size()[0]
                soft_loss = T**2 * self.criterion(student_logits/T, soft_targets)

                label_loss = self.criterion(student_logits, targets)
                # Weighted sum of the two losses
                loss = (1-b) * label_loss + b * soft_loss
                loss.backward()
                optimizer.step()

                summed_loss += loss.item()
                _, predicted = student_logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                _, teacher_predicted = teacher_logits.max(1)
                teacher_correct += teacher_predicted.eq(targets).sum().item()

            # Logging
            accuracy = 100. * correct / total
            teacher_accuracy = 100. * teacher_correct / total
            self.log_scalar("sigma_training.acc", accuracy)
            self.log_scalar("sigma_training.teacher_acc", teacher_accuracy)
        #record the learned sigma
        for p in self.model.named_parameters():
            if p[0].find('std')>=0:
                self.log_scalar("learned_sigma", p[1].data.item())
        # set back learnable params
        for p in self.model.named_parameters():
            #print(p[0], p[1].requires_grad)
            if p[0].find('weight')>=0 or p[0].find('bias')>=0 :
                p[1].requires_grad = True
            else:
                p[1].requires_grad = False

# We can call this command, e.g., from a Jupyter notebook with init_all=False to get an "empty" experiment wrapper,
# where we can then for instance load a pretrained model to inspect the performance.
@sacred_exp.command(unobserved=True)
def get_experiment(init_all=False):
    print('get_experiment')
    experiment = ExperimentWrapper(sacred_exp, init_all=init_all)
    return experiment


@sacred_exp.capture
def check_if_curr_exp_has_noise_during_training(noise_settings):
    # First check if no noise is applied during training
    noise_during_training = False
    # Check if global noise is applied during training
    if noise_settings['default']['noise_type'] != 'NoNoise':
        if noise_settings['default']['enable_in_training']:
            noise_during_training = True
    # Check if any layer wise noise is applied during training
    if noise_settings['layer_wise'] is not None:
        if noise_settings['layer_wise']['enable_in_training']:
            noise_during_training = True
    return noise_during_training

@sacred_exp.capture
def get_no_noise_db_entry_equivalent_to_current_exp(general, noise_settings, data, model, optimizer, db_collection, _log):
    # Load already computed seml results, which have no noise and check if an equivalent one exists
    # Select experiments, which have no default noise AND which have no layer noise
    filter_dict = {
        #'config.noise_settings.default.noise_type': 'NoNoise',
        'config.noise_settings.layer_wise': None,
        'config.noise_settings.layer_mapped': None,
    }
    fields_to_get = ['config', 'result']
    seml_res = seml.get_results(db_collection, filter_dict=filter_dict, fields=fields_to_get, to_data_frame=False)
    pre_trained_db_entry = None
    for completed_exp in seml_res:
        completed_exp_cfg = completed_exp['config']
        # Check if the experiment is actually without noise (these two operations should already be done by
        # MongoDB through the use of the filter_dict variable)
        if not completed_exp_cfg['noise_settings']['default']['noise_type'] == 'NoNoise' and "GaussStd_in_training" not in noise_settings["default"]:
            continue
        if completed_exp_cfg['noise_settings']['layer_wise'] is not None:
            continue
        # Check if the experiment matches our current config, excluding the noise
        # ToDo: Make this stricter again, i.e. completed_exp_cfg['model']['conf_name'] should check the whole model again, not just the conf
        # This was changed to make sure it dosen't take the repetition config into account, but it should take other stuff into account.
        # So it should do: completed_exp_cfg['model'] == model
        # But excluting the repetition_config key.
        ### stricter conditions by Xiao
        same_model = completed_exp_cfg['model'] == model
        same_general = completed_exp_cfg['general'] == general
        #if same_model is not True or same_general is not True:
        if same_model is not True:
            continue
        gauss_std_in_training = noise_settings["default"].get("GaussStd_in_training", 0)
        gotten_gauss_std_in_training = completed_exp_cfg['noise_settings']['default'].get("GaussStd_in_training", 0)
        if gauss_std_in_training != gotten_gauss_std_in_training:
            continue
        std_of_std_in_training = noise_settings["default"].get("Std_of_GaussStd_in_training", 0)
        gotten_std_of_std_in_training = completed_exp_cfg['noise_settings']['default'].get("Std_of_GaussStd_in_training", 0)
        if std_of_std_in_training != gotten_std_of_std_in_training:
            continue
        mean_of_std_in_training = noise_settings["default"].get("Mean_of_GaussStd_in_training", 0)
        gotten_mean_of_std_in_training = completed_exp_cfg['noise_settings']['default'].get("Mean_of_GaussStd_in_training", 0)
        if mean_of_std_in_training != gotten_mean_of_std_in_training:
            continue

        ### below is the loose conditions by Hendrik, could merged after check
        internal_sub_model = model[list(model.keys())[0]]['conf_name']
        gotten_sub_model = completed_exp_cfg['model'][list(completed_exp_cfg['model'].keys())[0]]['conf_name']
        # Same for the number of epochs, or rather the general key, here we wated to cut out the 'experiment_name' key.
        internal_num_epochs = general['num_epochs']
        gotten_num_epochs = completed_exp_cfg['general']['num_epochs']
        internal_repeat_number = general['repeat_number']
        gotten_repeat_number = completed_exp_cfg['general']['repeat_number']
        if completed_exp_cfg['data'] == data and gotten_sub_model == internal_sub_model and \
                completed_exp_cfg['optimizer'] == optimizer and gotten_num_epochs == internal_num_epochs  and gotten_repeat_number == internal_repeat_number:
            _log.info('Found an equivalent pretrained model')
            pre_trained_db_entry = completed_exp
            break
    return pre_trained_db_entry

@sacred_exp.capture
def load_checkpoint_from_exp(pre_trained_db_entry, _log, device='cpu'):
    _log.info(f'Loading model, with _id: {pre_trained_db_entry["_id"]} to device: {device}')
    # Load the checkpoint of the pre-trained model
    checkpoint_path = Path(pre_trained_db_entry['result']['artifacts']['Trained model checkpoint'])
    local_checkpoint_path = cluster.convert_artifact_path_to_local_path(checkpoint_path, logger=_log)
    if local_checkpoint_path.exists():
        pre_trained_checkpoint = torch.load(local_checkpoint_path, map_location=device)
        found_pre_trained_model = True
    else:
        found_pre_trained_model = False
        pre_trained_checkpoint = None
        _log.warning("Could not load the model, "
                     "because the checkpoint file doesn't exist on the local artifact storage.")
    return found_pre_trained_model, pre_trained_checkpoint


# In some cases the model being trained doesn't inject any noise during training, only during evaluation.
# In these cases the training is equivalent to noise less training, so we try to load the model from an equivalent
# training run.
@sacred_exp.capture
def get_pre_trained_checkpoint(general, data, model, optimizer, noise_settings, db_collection, _log, device='cpu'):
    # Check if the model can be loaded from a pre-existing checkpoint
    found_pre_trained_model = False
    pre_trained_checkpoint = None
    noise_during_training = check_if_curr_exp_has_noise_during_training(noise_settings)

    if not noise_during_training:
        _log.info('Found that this experiment contains no noise at training time.')
        _log.info('Searching for a pretrained model without noise.')
        pre_trained_db_entry = get_no_noise_db_entry_equivalent_to_current_exp(general, noise_settings, data, model, optimizer, db_collection, _log)
        #pre_trained_db_entry = None
        if pre_trained_db_entry is None:
            _log.warning('No equivalent pre-trained model was found, '
                         'evaluating this experiment may take longer than what would otherwise be required.')
        else:
            found_pre_trained_model, pre_trained_checkpoint = load_checkpoint_from_exp(pre_trained_db_entry, _log, device=device)

    return found_pre_trained_model, pre_trained_checkpoint


def get_free_gpus(_log):
    """
    Checks nvidia-smi for available GPUs and returns those with the most available memory.
    Inspired by: https://discuss.pytorch.org/t/it-there-anyway-to-let-program-select-free-gpu-automatically/17560/2
    """
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]).decode()
    gpu_stats = gpu_stats.replace(' MiB', '')
    gpu_df = pd.read_csv(StringIO(gpu_stats),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    _log.info('GPU usage [MiB]:\n{}'.format(gpu_df))
    valid_gpus = gpu_df.loc[gpu_df['memory.free'] == gpu_df['memory.free'].max()].index.values
    _log.info(f'GPUs with the most available memory: {list(valid_gpus)}')
    return valid_gpus


# This function will be called by default. Note that we could in principle manually pass an experiment instance,
# e.g., obtained by loading a model from the database or by calling this from a Jupyter notebook.
@sacred_exp.automain
def train(general, model, noise_settings, _log, experiment=None):
    # If we are running on a GPU, then select the device with the most available memory
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        # Wait for a random amount of time before initializing
        # Only sleep if there is more than one GPU available
        avail_devices = get_free_gpus(_log)
        if len(avail_devices) > 1:
            sleep_delay = np.random.randint(0, 60)
            _log.info(f'Multiple GPUs available: Sleeping for {sleep_delay} seconds before starting.')
            time.sleep(sleep_delay)
        else:
            _log.info("Only one GPU available: Skipping sleep")
        # Get the GPUs with the most memory and select one randomly
        avail_devices = get_free_gpus(_log)
        device = int(np.random.choice(avail_devices))
        _log.info(f'Selected GPU: {device}')
    # Exception for octane005: Force CPU usage
    if os.uname().nodename == 'octane005.ziti.uni-heidelberg.de':
        _log.info("Detected host as octane005. "
                  "Forcing 'cpu' as torch device to avoid issues with the unregistered GPU on this system.")
        device = 'cpu'
    device = torch.device(device)

    # Create the experiment wrapper
    if experiment is None:
        experiment = ExperimentWrapper(sacred_exp, device=device)
    
    # Check if we can get a pre-trained checkpoint for this particular experiment
    # This usually reduces the compute time immensely
    # Note that all the required arguments get inserted by sacred automatically
    found_pre_trained_model, pre_trained_checkpoint = get_pre_trained_checkpoint(device=device)
    if found_pre_trained_model:
        experiment.model.load_state_dict(pre_trained_checkpoint['state_dict'], strict=False)
        # Note from Xiao: after modifying NoiseOperator as nn.Module and taking GaussStd as nn.Parameter, \
        # loading state dict from previous model will also load the noisy layer with previous std, \
        # so should manually reset the noisy layer as the current std.
        cur_std = 0
        if noise_settings["default"]["noise_type"] != "NoNoise":
            cur_std = noise_settings["default"]["GaussStd"]
        noise_new = copy.deepcopy(noise_settings["default"])
        noise_new["GaussStd"] = cur_std
        noise_config = cfg.resolve_config_from_name(noise_new["noise_type"], **noise_new)
        noise_case = ReConfigNoise(noise_config)
        experiment.model.apply(noise_case).to(device)
        _log.info(f"Load model from pre_trained_model and reset the std as: {cur_std}")
        #_log.info(f"Load model from pre_trained_model: {experiment.model}")
    # Training
    start_time = time.time()
    if found_pre_trained_model:
        # The model is already trained, so we simply do one validation step to compute the accuracy with noise
        val_loss, val_acc = experiment.validation_step()
        # Commandline logging
        status_dict = {
            "Validation: Loss": val_loss,
            "v_Accuracy": val_acc,
        }

        # Optional test dataset for some dataset
        if experiment.data.has_test_dataset:
            test_loss, test_acc = experiment.validation_step(run_on_test_dataset_instead=True)
            status_dict['Test: Loss'] = test_loss
            status_dict['ts_Accuracy'] = test_acc

        print_status(_log, start_time, 0, 1, **status_dict)
    else:
        # Do some actual training
        # if it is incremental training
        std_list = []
        noise_info = noise_settings["default"]
        # Done: Extend to support layer wise noise
        noise_default_type = noise_settings["default"]["noise_type"]
        if noise_default_type != "NoNoise":
            inf_std = noise_settings["default"]["GaussStd"]
            # global noise, then three possibility
            # 1. enable_in_training --> ideal noisy training
            # 2. enable_in_training is false, but didn't find the pretrained model, 
            #    then default training without noise injection 
            # 3. or training with a specified std 
            if noise_settings["default"]["enable_in_training"]:
                train_std = noise_settings["default"]["GaussStd"]
            elif "GaussStd_in_training" in noise_settings["default"]:
                train_std = noise_settings["default"]["GaussStd_in_training"]
            else:
                train_std = 0
        elif noise_settings["layer_wise"] is not None:
            # layer wise noise
            train_std = noise_settings["layer_wise"]["GaussStd"]
            noise_info = noise_settings["layer_wise"]
        else:
            # global nonoise training
            train_std = 0
        if "is_incremental" in noise_settings and noise_settings["is_incremental"] and train_std > 0:
            # Done: Modify linspace, such that maximum noise is reached at about 50 to 20 epochs before the training ends
            inc_list = np.linspace(0, train_std, 100)
            max_list = [train_std] * (general['num_epochs'] - 100)
            std_list = np.concatenate((inc_list, np.array(max_list)))
        if "is_overshooting" in noise_settings and noise_settings["is_overshooting"]and train_std > 0:
            overshoot_value = train_std * 1.5
            inc_list = np.linspace(0, overshoot_value, general['num_epochs'] - 175)
            dec_list = np.linspace(overshoot_value, train_std, 75)
            max_list = [train_std] * 100
            std_list = np.concatenate((inc_list, dec_list, np.array(max_list)))
        if "is_std_of_std" in noise_settings and noise_settings["is_std_of_std"] and not noise_settings.get("enable_std_per_img", False):
            mean_of_std = train_std
            if "is_mean_of_std" in noise_settings and noise_settings["is_mean_of_std"]:
                mean_of_std = train_std * noise_settings["default"]["Mean_of_GaussStd_in_training"]
            std_of_std = noise_settings["default"]["Std_of_GaussStd_in_training"]
            #std_list = list(mean_of_std + std_of_std * mean_of_std * np.abs(np.random.randn(general['num_epochs'])))
            std_list = list(mean_of_std + std_of_std * np.abs(np.random.randn(general['num_epochs'])))
        if "is_noisyweight" in noise_settings and noise_settings["is_noisyweight"]:
            # remove the noise injection on activations during training
            std_list = [0] * general['num_epochs']
        if "param_sigma" in noise_settings and noise_settings["param_sigma"]:
            # learn the sigmas
            experiment.training_param_sigmas()
        for epoch in range(general['num_epochs']):
            # if incremental training, then reset the noise strength
            cur_train_std = 0
            if len(std_list) == general['num_epochs']:
                cur_train_std = std_list[epoch]
            elif train_std > 0:
                cur_train_std = train_std
            if cur_train_std != 0:
                noise_new = copy.deepcopy(noise_info)
                noise_new["GaussStd"] = cur_train_std
                noise_config = cfg.resolve_config_from_name(noise_new["noise_type"], **noise_new)
                noise_case = ReConfigNoise(noise_config)
                experiment.model.apply(noise_case).to(device)
                experiment.log_scalar("train_std",  cur_train_std)
            # Training
            if "enable_awp" in general and general["enable_awp"]:
                train_loss, train_acc = experiment.training_step_awp()
            elif "adv_type" in general and general["adv_type"] != "NoAdv":
                train_loss, train_acc = experiment.training_step_adv()
            elif "distill" in model and model["distill"]["is_enabled"]:
                train_loss, train_acc = experiment.training_step_distill(model)
            elif "enable_sam" in general and general["enable_sam"]:
                train_loss, train_acc = experiment.training_step_sam()
            elif "is_noisyweight" in noise_settings and noise_settings["is_noisyweight"]:
                train_loss, train_acc = experiment.training_step_noisy_weight()
            else:
                train_loss, train_acc = experiment.training_step(cur_train_std=cur_train_std)
            # Validation using GaussStd
            if noise_default_type != "NoNoise": 
                #if inf_std == cur_train_std and enable_std_per_img is on, then also should reset the inf_std as a single variable
                noise_new = copy.deepcopy(noise_info)
                noise_new["GaussStd"] = inf_std
                noise_config = cfg.resolve_config_from_name(noise_new["noise_type"], **noise_new)
                noise_case = ReConfigNoise(noise_config)
                # if the the std is learnable parameter, i.e. "param_sigma" in noise_setting, then should backup the model before model.apply
                backup_model = copy.deepcopy(experiment.model)
                # prepare the model for inference
                experiment.model.apply(noise_case).to(device)
                experiment.log_scalar("inf_std",  inf_std)
                val_loss, val_acc = experiment.validation_step()
                #set back to the model paremeters
                if "param_sigma" in noise_settings and noise_settings["param_sigma"]:
                    experiment.model = copy.deepcopy(backup_model)
            else:
                val_loss, val_acc = experiment.validation_step()
            experiment.scheduler.step()

            # Commandline logging
            status_dict = {
                "Training: Loss": train_loss,
                "tr_Accuracy:": train_acc,
                "Validation: Loss": val_loss,
                "v_Accuracy": val_acc,
            }

            # Optional test dataset for some dataset
            if experiment.data.has_test_dataset:
                test_loss, test_acc = experiment.validation_step(run_on_test_dataset_instead=True)
                status_dict['Test: Loss'] = test_loss
                status_dict['ts_Accuracy'] = test_acc

            print_status(_log, start_time, epoch, general['num_epochs'], **status_dict)

        # Export and save model data
        # before save models, reset the std of each activations as normal shape
        config_hash = experiment.get_config_hash()
        export_path = f"{config_hash}_{str(uuid.uuid4())}_trained_model.pt"
        torch.save({
            'state_dict': experiment.model.state_dict(),
            'optim_dict': experiment.optimizer.state_dict(),
            'scheduler_dict': experiment.scheduler.state_dict(),
        }, export_path)
        # Wait a bit to make sure the file is completely written before it gets uploaded
        time.sleep(10.)
        experiment.add_artifact(export_path, 'Trained model checkpoint')
        # Again, wait a bit to make sure the upload completes
        time.sleep(10.)
        # Make sure the artifact gets removed from the local disk
        os.remove(export_path)

    # Save the result data with seml
    return experiment.get_seml_return_data()
