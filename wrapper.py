from configurations import cmd_args
from model_run_iterative import prune_and_train as prune_and_train_iterative
from model_run import prune_and_train as prune_and_train_one_shot

if __name__ == "__main__":
    if cmd_args["node_based"]:
        prune_and_train_one_shot()
    if cmd_args["is_iterative"]:
        steps = [0.1, 0.2, 0.31]
        epoch_list = [500, 500, 1500]
        prune_and_train_iterative(steps, epoch_list)
    else:
        prune_and_train_one_shot()
