import argparse


def get_argparser():
    argparser = argparse.ArgumentParser(description="put parameters")
    argparser.add_argument("--pruning_amount", type=float, default=0.9, help="")
    argparser.add_argument("--run_netron", type=bool, default=False, help="")
    argparser.add_argument(
        "--pruning_mode", type=str, default="structured", choices=["structured", "SIMD"]
    )
    return argparser.parse_args()


argparser = get_argparser()
pruning_amount = argparser.pruning_amount
run_netron = argparser.run_netron
pruning_mode = argparser.pruning_mode
