import os
import torch
import time
import torch.optim as optim
import qonnx.core.onnx_exec as oxe
from imports import (
    TrainingEpochMeters,
    SqrHingeLoss,
    weight_decay,
    accuracy,
    lr,
    prune_brevitas_model,
    prune_brevitas_modelSIMD,
    log_freq,
    EvalEpochMeters,
    eval_model,
)
import torch.nn as nn
import datetime
from main import showInNetron
from qonnx.core.modelwrapper import ModelWrapper
from torch.utils.data import DataLoader
from torchvision import transforms
from brevitas.export import export_qonnx
from torchvision.datasets import MNIST, CIFAR10
from imports import get_test_model_trained
import argparse
import netron
from IPython.display import IFrame
import onnx.numpy_helper as numpy_helper


def get_argparser():
    argparser = argparse.ArgumentParser(description="put parameters")
    argparser.add_argument("--pruning_amount", type=float, default=0.4, help="")
    return argparser.parse_args()


argparser = get_argparser()
pruning_amount = argparser.pruning_amount

build_dir = "models_folder"
datadir = "./data/"
dataset = "CIFAR10"
epoch_data = {"train": {}, "test": {}}
epoch_data["train"][str(pruning_amount)] = []
epoch_data["test"][str(pruning_amount)] = []
num_classes = 10
num_workers = 0
batch_size = 100
file1 = open(f"pruning_logs_{str(pruning_amount)}.txt", "a")
now = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
file1.write(f"Starting to write at {now}\nPruning Amount: {pruning_amount}")
file1.flush()
if True or dataset == "CIFAR10":
    pass
transform_to_tensor = transforms.Compose([transforms.ToTensor()])
train_transforms_list = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
]
transform_train = transforms.Compose(train_transforms_list)
builder = CIFAR10

export_onnx_path = build_dir + "/end2end_cnv_w1a1_export_to_download.onnx"
showInNetron(export_onnx_path)
model = ModelWrapper(export_onnx_path)
model = get_test_model_trained("CNV", 1, 1)
prune_brevitas_model(model, conv_feature_index=8, SIMD=32,NumColPruned=)
prune_brevitas_model(model, conv_feature_index=11, SIMD=32,NumColPruned=)
prune_brevitas_model(model, conv_feature_index=15, SIMD=32,NumColPruned=)
prune_brevitas_model(model, conv_feature_index=18, SIMD=32,NumColPruned=)
# model = CNV(10, WEIGHT_BIT_WIDTH, ACT_BIT_WIDTH, 8, 3).to(device=device)
eval_meters = EvalEpochMeters()
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

criterion = SqrHingeLoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = criterion.to(device=device)
starting_epoch = 1
best_val_acc = 0
epochs = 30
for epoch in range(starting_epoch, epochs):
    # Set to training mode
    model.train()
    criterion.train()

    # Init metrics
    epoch_meters = TrainingEpochMeters()
    start_data_loading = time.time()

    for i, data in enumerate(train_loader):
        (input, target) = data
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # for hingeloss only
        if isinstance(criterion, SqrHingeLoss):
            target = target.unsqueeze(1)
            target_onehot = torch.Tensor(target.size(0), num_classes).to(
                device, non_blocking=True
            )
            target_onehot.fill_(-1)
            target_onehot.scatter_(1, target, 1)
            target = target.squeeze()
            target_var = target_onehot
        else:
            target_var = target

        # measure data loading time
        epoch_meters.data_time.update(time.time() - start_data_loading)

        # Training batch starts
        start_batch = time.time()
        output = model(input)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.clip_weights(-1, 1)

        # measure elapsed time
        epoch_meters.batch_time.update(time.time() - start_batch)
        pred = output.data.argmax(1, keepdim=True)
        correct = pred.eq(target.data.view_as(pred)).sum()
        prec1 = 100.0 * correct.float() / input.size(0)
        if i % int(log_freq) == 0:
            file1.write(f"Epoch: {epoch} Batch: {i} accuracy {str(prec1)}\n")
            file1.flush()

        eval_meters.top1.update(prec1.item(), input.size(0))

    # Perform eval
    with torch.no_grad():
        top1avg, save_data_list = eval_model(
            model, criterion, test_loader, num_classes, epoch, device
        )
    epoch_data["test"][str(pruning_amount)].append(top1avg)
    epoch_data["train"][str(pruning_amount)].append(eval_meters.top1.avg)
    file1.write(
        f"Epoch {epoch} complete. Train  accuracy {str(eval_meters.top1.avg)}\n"
    )
    file1.flush()
    file1.write(f"Epoch {epoch} complete. Test accuracy {str(top1avg)}\n")
    file1.flush()
    # checkpoint
    # Skip the actual saving as it uses up too much data
    if top1avg >= best_val_acc:
        best_val_acc = top1avg
        # checkpoint_best(epoch, f"best_pruning_amount-{pruning_amount:.3f}.tar")
    else:
        pass
        # checkpoint_best(epoch, f"checkpoint_pruning_amount-{pruning_amount:.3f}.tar")

    # writing newline character
