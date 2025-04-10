import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Function
import os
import datetime


def get_argparser():
    argparser = argparse.ArgumentParser(description="put parameters")
    argparser.add_argument("--pruning_amount", type=float, default=0.9, help="")
    argparser.add_argument("--run_netron", type=bool, default=False, help="")
    argparser.add_argument("--model", type=str, default="cnv_1w1a", help="")
    argparser.add_argument(
        "--pruning_mode", type=str, default="structured", choices=["structured", "SIMD"]
    )
    return argparser.parse_args()


argparser = get_argparser()
# pruning_amount = argparser.pruning_amount
pruning_amount = [0.0] * 4 + [0.5] * 4 + [0.0] * 1
run_netron = argparser.run_netron
pruning_mode = argparser.pruning_mode
model_identity = argparser.model
model_identity = "cnv_4w4a"

now_time = datetime.datetime.now()
now_str = now_time.strftime("%d_%b_%Y__%H_%M_%S")
pruning_type = f"{pruning_mode}_{model_identity}"
if not os.path.exists(f"runs/{pruning_type}"):
    os.mkdir(f"runs/{pruning_type}")
os.mkdir(f"runs/{pruning_type}/{now_str}")
path_for_save = f"runs/{pruning_type}/{now_str}"

network = "cnv"
experiments = "."
datadir = "./data/"
batch_size = 100
num_workers = 6
lr = 0.03
lr_schedule_period = 40
lr_schedule_ratio = 0.5
weight_decay = 0
random_seed = 1
log_freq = 10
epochs, num_classes, starting_epoch, best_val_acc = 500, 10, 0, 0

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
