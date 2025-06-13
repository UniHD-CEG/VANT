# source:https://github.com/BayesWatch/cinic-10?tab=readme-ov-file#data-loading 
'''
cinic_directory = '/path/to/cinic/directory'
cinic_mean = [0.47889522, 0.47227842, 0.43047404]
cinic_std = [0.24205776, 0.23828046, 0.25874835]
cinic_train = torch.utils.data.DataLoader(
    torchvision.datasets.ImageFolder(cinic_directory + '/train',
    	transform=transforms.Compose([transforms.ToTensor(),
        transforms.Normalize(mean=cinic_mean,std=cinic_std)])),
    batch_size=128, shuffle=True)
'''
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import os

class CINIC10():
    name = "CINIC10"
    dims = (3, 32, 32)
    has_test_dataset = False

    def __init__(self, batch_size=128, num_workers=2, data_root="/tmp/datasets/"):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean,std=cinic_std),
        ])

        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cinic_mean,std=cinic_std),
        ])

    @property
    def num_classes(self) -> int:
        """
        Return:
            10
        """
        return 10

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        self.train_set = datasets.ImageFolder(root=os.path.join(self.data_root, 'CINIC-10', 'train'), transform=self.transform_train)
        self.val_set = datasets.ImageFolder(root=os.path.join(self.data_root, 'CINIC-10', 'valid'), transform=self.transform_val)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_set, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_set, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers)
        return val_loader

