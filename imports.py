import torch
import json
import brevitas.config as config
from brevitas.nn import QuantConv2d

# from brevitas.core.restrict_val import RestrictValueType
import time
import torch_pruning as tp
import matplotlib.pyplot as plt

# from torch_pruning import utils
from brevitas.export import export_qonnx
from visualize_netron import showInNetron
from configurations import (
    now_time,
    pruning_amount,
    pruning_mode,
    SqrHingeLoss,
    model_identity,
    weight_decay,
    lr,
    lr_schedule_period,
    lr_schedule_ratio,
)

example_inputs = torch.randn(1, 3, 32, 32)
SIMD_LIST = [3, 32, 32, 32, 32, 32, 32, 32, 64]


def disable_jit(func):
    def wrapper(*args, **kwargs):
        config.JIT_ENABLED = 0
        result = func(*args, **kwargs)
        config.JIT_ENABLED = 1
        return result

    return wrapper


@disable_jit
def prune_wrapper(model, pruning_amount, pruning_mode, run_netron, folder_name):
    onnx_path_extended = f"{folder_name}/extended_model"

    export_qonnx(
        model,
        args=example_inputs.cpu(),
        export_path=f"{onnx_path_extended}.onnx",
        opset_version=13,
    )
    pruning_data = prune_all_conv_layers(
        model,
        SIMD_list=SIMD_LIST,
        NumColPruned=pruning_amount,
        pruning_mode=pruning_mode,
    )
    pruned_onnx_filename = f"{onnx_path_extended}_pruned"
    export_qonnx(
        model,
        args=example_inputs.cpu(),
        export_path=f"{pruned_onnx_filename}.onnx",
        opset_version=13,
    )
    if run_netron:
        showInNetron(f"{onnx_path_extended}.onnx", port=8080)
        showInNetron(f"{pruned_onnx_filename}.onnx", port=8082)
    with open(f"{pruned_onnx_filename}.json", "w") as fp:
        fp.write(json.dumps(pruning_data, indent=4, ensure_ascii=False))
    config.JIT_ENABLED = 1
    return model


def conv_layer_traverse(model):
    for layer in model.conv_features:
        if isinstance(layer, QuantConv2d):
            yield layer


def weight_histograms(model, folder_name):
    length = sum(1 for _ in conv_layer_traverse(model))
    fig, axs = plt.subplots(length)
    for layer_idx, layer in enumerate(conv_layer_traverse(model)):
        axs[layer_idx].hist(get_layer_tensor(layer.weight.data))
    plt.savefig(f"./{folder_name}/weight_hist.png")


def prune_all_conv_layers(model, SIMD_list, NumColPruned=-1, pruning_mode="structured"):
    pruning_data = []
    for layer_idx, layer in enumerate(conv_layer_traverse(model)):
        if layer_idx == 0:
            print("Initial layer skipped as channels are RGB channels.")
            continue
        SIMD = SIMD_list[layer_idx]
        in_channels = layer.in_channels
        try:
            assert (
                in_channels % SIMD == 0
            ), f"SIMD must divide IFM Channels. Pruning {layer} is skipped."
        except AssertionError:
            continue
        pruning_ratio = (
            NumColPruned[layer_idx] if isinstance(NumColPruned, list) else NumColPruned
        )
        if pruning_mode == "structured":
            pruning_entities = prune_brevitas_model(
                model, layer_to_prune=layer, SIMD=SIMD, NumColPruned=pruning_ratio
            )
        else:
            pruning_entities = prune_brevitas_modelSIMD(
                model, layer_to_prune=layer, SIMD_in=SIMD, NumColPruned=pruning_ratio
            )
        pruning_data.append(
            {
                "pruned_layer_index": layer_idx,
                "pruning_mode": pruning_mode,
                "pruning_entities": pruning_entities,
            }
        )
    return pruning_data


def sort_tensor(tensor):
    x = tensor.permute(1, 0, 2, 3)
    x = x.reshape(tensor.shape[1], -1)
    out = x.abs().sum(dim=1, keepdim=True)
    sorted_indices = out.argsort(dim=0)
    return [i.item() for i in sorted_indices.squeeze()]


def get_layer_tensor(tensor):
    x = tensor.permute(1, 0, 2, 3)
    x = x.reshape(tensor.shape[1], -1)
    out = x.abs().sum(dim=1, keepdim=True)
    return [i.item() for i in out]


def prune_brevitas_modelSIMD(
    model, layer_to_prune, SIMD_in=1, NumColPruned=-1, SIMD_out=-1
) -> dict:
    in_channels = layer_to_prune.in_channels
    assert in_channels % SIMD_in == 0, "SIMD must divide IFM Channels"
    num_of_blocks = int(in_channels / SIMD_in)
    if SIMD_out == -1:
        if isinstance(NumColPruned, float):
            SIMD_out = int(round(SIMD_in * (1.0 - NumColPruned)))
        if SIMD_out == 0:
            SIMD_out = 1
    in_channels_new = num_of_blocks * SIMD_out
    # channels_to_prune = math.floor(model.conv_features[conv_feature_index].in_channels * pruning_amount)
    sorting_indices = sort_tensor(layer_to_prune.weight.data)
    channels_to_prune = sorting_indices[: (-1) * in_channels_new]
    dep_graph = tp.DependencyGraph().build_dependency(
        model, example_inputs=example_inputs
    )
    group = dep_graph.get_pruning_group(
        layer_to_prune,
        tp.prune_conv_in_channels,
        idxs=channels_to_prune,
    )
    if dep_graph:
        group.prune()
    print(layer_to_prune.in_channels)
    return {
        "in_channels_old": in_channels,
        "in_channels_new": layer_to_prune.in_channels,
        "SIMD_in": SIMD_in,
        "SIMD_out": SIMD_out,
    }


def prune_brevitas_model(model, layer_to_prune, SIMD=1, NumColPruned=-1) -> dict:
    in_channels = layer_to_prune.in_channels
    print(f"in_channels={in_channels} SIMD={SIMD}")
    if isinstance(NumColPruned, float):
        NumColPruned = int(round((in_channels / SIMD) * NumColPruned))
    # channels_to_prune = math.floor(model.conv_features[conv_feature_index].in_channels * pruning_amount)
    prune_block_len = (
        SIMD * NumColPruned if SIMD * NumColPruned < in_channels else in_channels - SIMD
    )
    sorting_indices = sort_tensor(layer_to_prune.weight.data)
    channels_to_prune = sorting_indices[:prune_block_len]
    # channels_to_prune = [i for i in range(prune_block_len)]
    dep_graph = tp.DependencyGraph().build_dependency(
        model, example_inputs=example_inputs
    )
    group = dep_graph.get_pruning_group(
        layer_to_prune,
        tp.prune_conv_in_channels,
        idxs=channels_to_prune,
    )
    # group2 = dep_graph.get_pruning_group(model.conv_features[15], tp.prune_conv_in_channels, idxs=[0])
    # print(group.details())
    if dep_graph:
        group.prune()
    # utils.draw_groups(dep_graph, 'groups')
    # utils.draw_dependency_graph(dep_graph, 'dep_graph')
    # utils.draw_computational_graph(dep_graph,'comp_graph')
    return {
        "in_channels_old": in_channels,
        "in_channels_new": layer_to_prune.in_channels,
    }

    importance = tp.importance.GroupNormImportance(p=2, group_reduction="first")
    pruner = tp.pruner.GroupNormPruner(
        model,
        example_inputs=example_inputs,
        importance=importance,
        iterative_steps=1,
        pruning_ratio=0.0,
        global_pruning=False,
        round_to=8,
        pruning_ratio_dict={model.conv_features[1]: 0.1},
        root_module_types=[QuantConv2d],
        unwrapped_parameters=[[model.conv_features[1].weight, 0]],
        # customized_pruners={QuantConv2d: tp.pruner.prune_conv_in_channels}
    )

    for g in pruner.step(interactive=True):
        g.prune()
    if isinstance(
        pruner,
        (
            tp.pruner.BNScalePruner,
            tp.pruner.GroupNormPruner,
            tp.pruner.GrowingRegPruner,
        ),
    ):
        pruner.update_regularizer()  # if the model has been pruned, we need to update the regularizer
        pruner.regularize(model)


def save_best_checkpoint(best_model, optimizer, epoch, best_val_acc, best_path):
    torch.save(
        {
            "state_dict": best_model.state_dict(),
            "optim_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_val_acc": best_val_acc,
        },
        best_path,
    )


@disable_jit
def export_best_onnx(best_model, example_inputs, export_path):
    export_qonnx(
        best_model,
        example_inputs,
        export_path=export_path,
    )


class TrainingEpochMeters(object):
    def __init__(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    # return indices of larges k values
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].flatten().float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class EvalEpochMeters(object):
    def __init__(self):
        self.model_time = AverageMeter()
        self.loss_time = AverageMeter()
        self.losses = AverageMeter()
        self.top1 = AverageMeter()
        self.top5 = AverageMeter()


def eval_model(model, criterion, test_loader, num_classes=10, epoch=-1, device="cpu"):
    eval_meters = EvalEpochMeters()

    # switch to evaluate mode
    model.eval()
    criterion.eval()
    save_data_list = []

    for i, data in enumerate(test_loader):
        end = time.time()
        (inp, target) = data

        inp = inp.to(device, non_blocking=True)
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

        # compute output
        output = model(inp)

        # measure model elapsed time
        eval_meters.model_time.update(time.time() - end)
        end = time.time()

        # compute loss
        loss = criterion(output, target_var)
        eval_meters.loss_time.update(time.time() - end)

        pred = output.data.argmax(1, keepdim=True)
        correct = pred.eq(target.data.view_as(pred)).sum()
        prec1 = 100.0 * correct.float() / inp.size(0)

        _, prec5 = accuracy(output, target, topk=(1, 5))
        eval_meters.losses.update(loss.item(), inp.size(0))
        eval_meters.top1.update(prec1.item(), inp.size(0))
        eval_meters.top5.update(prec5.item(), inp.size(0))

        # Compile save data
        save_data = [time.time(), epoch, loss.item(), prec1.item(), prec5.item()]
        save_data_list.append(save_data)

    return eval_meters.top1.avg, save_data_list


def log_to_file(file, text):
    file.write(text + "\n")
    file.flush()


def start_log_to_file(path):
    file1 = open(f"{path}/pruning_logs.txt", "a")
    log_str = """Starting to write at {}
Pruning Amount: {}
Pruning Mode: {}
Model_Identity: {}
Weight_decay: {}
LR: {}
LR schedule period: {}
LR schedule ratio: {}
SIMD_LIST: {}
    """.format(
        now_time.strftime("%H:%M:%S%p on %d %B %Y"),
        pruning_amount,
        pruning_mode,
        model_identity,
        weight_decay,
        lr,
        lr_schedule_period,
        lr_schedule_ratio,
        SIMD_LIST,
    )
    print(log_str)
    log_to_file(file1, log_str)
    return file1
