import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Function
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    MultiStepLR,
)
import os
import datetime
import shutil


def get_argparser():
    argparser = argparse.ArgumentParser(description="put parameters")
    argparser.add_argument("--pruning_amount", type=float, default=0.9, help="")
    argparser.add_argument("--run_netron", type=bool, default=False, help="")
    argparser.add_argument("--use_scheduler", type=bool, default=True, help="")
    argparser.add_argument("--model", type=str, default="cnv_1w1a", help="")
    argparser.add_argument(
        "--pruning_mode", type=str, default="structured", choices=["structured", "SIMD"]
    )
    return argparser.parse_args()


argparser = get_argparser()
# pruning_amount = argparser.pruning_amount
pruning_amount = [0.5] * 4 + [0.5] * 4 + [0.5] * 1
run_netron = argparser.run_netron
pruning_mode = argparser.pruning_mode
use_scheduler = argparser.use_scheduler
model_identity = argparser.model

now_time = datetime.datetime.now()
now_str = now_time.strftime("%d_%b_%Y__%H_%M_%S")
pruning_type = f"{pruning_mode}_{model_identity}"
if not os.path.exists(f"runs/{pruning_type}"):
    os.mkdir(f"runs/{pruning_type}")
os.mkdir(f"runs/{pruning_type}/{now_str}")
path_for_save = f"runs/{pruning_type}/{now_str}"
shutil.copyfile("./configurations.py", f"{path_for_save}/configurations.py")
shutil.copyfile("./model_run.py", f"{path_for_save}/model_run.py")
network = "cnv"
experiments = "."
datadir = "./data/"
batch_size = 400
num_workers = 6
lr = 0.01
lr_schedule_period = 30
lr_schedule_ratio = 0.5
eta_min = lr * (0.5**7)
T_max = 50
T_mult = 2
weight_decay = 0
random_seed = 1
log_freq = 10
epochs, num_classes, starting_epoch, best_val_acc = 1000, 10, 0, 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class squared_hinge_loss(Function):
    @staticmethod
    def forward(ctx, predictions, targets):
        ctx.save_for_backward(predictions, targets)
        output = 1.0 - predictions.mul(targets)
        output[output.le(0.0)] = 0.0
        loss = torch.mean(output.mul(output))
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        predictions, targets = ctx.saved_tensors
        output = 1.0 - predictions.mul(targets)
        output[output.le(0.0)] = 0.0
        grad_output.resize_as_(predictions).copy_(targets).mul_(-2.0).mul_(output)
        grad_output.mul_(output.ne(0).float())
        grad_output.div_(predictions.numel())
        return grad_output, None


class SqrHingeLoss(nn.Module):
    # Squared Hinge Loss
    def __init__(self):
        super(SqrHingeLoss, self).__init__()

    def forward(self, input, target):
        return squared_hinge_loss.apply(input, target)


def get_optimizer(model):
    criterion = SqrHingeLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = criterion.to(device=device)
    return criterion, optimizer


def get_scheduler(optimizer, T_max, eta_min, T_mult=1):
    #return CosineAnnealingLR(optimizer=optimizer, T_max=T_max, eta_min=eta_min)
    return CosineAnnealingWarmRestarts(
       optimizer=optimizer, T_0=T_max, T_mult=T_mult, eta_min=eta_min
    )
    #return MultiStepLR(
    #    optimizer=optimizer, milestones=[30, 80, 140, 200, 260], gamma=0.5
    #)
