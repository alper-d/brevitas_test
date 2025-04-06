import argparse
import torch
import torch.optim as optim
from imports import SqrHingeLoss
import os


def get_argparser():
    argparser = argparse.ArgumentParser(description="put parameters")
    argparser.add_argument("--pruning_amount", type=float, default=0.9, help="")
    argparser.add_argument("--run_netron", type=bool, default=False, help="")
    argparser.add_argument("--model", type=str, default="cnv_1w1a", help="")
    argparser.add_argument(
        "--pruning_mode", type=str, default="structured", choices=["structured", "SIMD"]
    )
    return argparser.parse_args()


os.environ["BREVITAS_JIT"] = "1"
argparser = get_argparser()
# pruning_amount = argparser.pruning_amount
pruning_amount = [0.0, 0.0] + [0.5] * 4 + [0.0] * 3
run_netron = argparser.run_netron
pruning_mode = argparser.pruning_mode
model_identity = argparser.model
model_identity = "cnv_1w2a"

network = "cnv"
experiments = "."
datadir = "./data/"
batch_size = 100
num_workers = 6
lr = 0.02
weight_decay = 0
random_seed = 1
log_freq = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_optimizer(model):
    criterion = SqrHingeLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    criterion = criterion.to(device=device)
    return criterion, optimizer
