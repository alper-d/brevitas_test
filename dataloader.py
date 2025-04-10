from torchvision.datasets import CIFAR10
import random
import torch

# from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

builder = CIFAR10
datadir = "./data/"
dataset = "CIFAR10"
num_workers = 0
batch_size = 100

if True or dataset == "CIFAR10":
    pass
transform_to_tensor = transforms.Compose([transforms.ToTensor()])
train_transforms_list = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]
transform_train = transforms.Compose(train_transforms_list)
train_set = builder(root=datadir, train=True, download=True, transform=transform_train)
test_set = builder(
    root=datadir, train=False, download=True, transform=transform_to_tensor
)
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
)
test_loader = DataLoader(
    test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
)
