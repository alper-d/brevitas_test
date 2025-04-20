from configurations import cmd_args
from model_run_iterative import prune_and_train

if __name__ == "__main__":
    if cmd_args["is_iterative"]:
        steps = [0.1, 0.2, 0.3]
        prune_and_train(steps)
