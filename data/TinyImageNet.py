# source: http://cs231n.stanford.edu/tiny-imagenet-200.zip
# 100,000 images of 200 classes downsized to 64*64 colored images from ImageNet.
# Each class has 500 training images, 50 validation images and 50 test images.

from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
import os

class TinyImageNet():
    name = "TinyImageNet"
    dims = (3, 64, 64)
    has_test_dataset = False

    def __init__(self, batch_size=128, num_workers=2, data_root="/tmp/datasets/"):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_root = data_root

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    @property
    def num_classes(self) -> int:
        """
        Return:
            200
        """
        return 200

    def prepare_data(self) -> None:
        pass

    def setup(self) -> None:
        self.train_set = datasets.ImageFolder(root=os.path.join(self.data_root, 'tiny-imagenet-200', 'train'), transform=self.transform_train)
        self.val_set = datasets.ImageFolder(root=os.path.join(self.data_root, 'tiny-imagenet-200', 'val'), transform=self.transform_val)

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

