from typing import Optional, Tuple
import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision import transforms

class MNISTDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = './data',
        train_val_split: Tuple[float, float] = [.9, .1],
        batch_size: int = 250, #how to automatically calculate batch size based on GPU mem?
    ) -> None:
        """Initialize a `LightningDataModule` for training on MNIST.

        :param dataset_path: path to the MNIST dataset
        :param train_val_split: The train, val, test split. Defaults to `(0.9, 0.1)`.
        :param batch_size: The batch size. Defaults to `250`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.mnist_train: Optional[Dataset] = None
        self.mnist_val: Optional[Dataset] = None
        self.mnist_test: Optional[Dataset] = None

        #transform to tensor and normalize images
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)), 
            transforms.ToTensor(), 
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.data_dir = data_dir
        self.batch_size = batch_size

    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage):
        #fetch training data
        if stage == "fit":
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, self.hparams.train_val_split, generator=torch.Generator().manual_seed(42)
            )
        
        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size, num_workers=4)
