import os

import torch.nn as nn
import torch
from torch.autograd import Function
import json
from torch.nn import Module, ModuleList, BatchNorm2d, MaxPool2d, BatchNorm1d
import brevitas.config as config
from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear

# from brevitas.core.restrict_val import RestrictValueType
from dependencies import value
import torch.nn.init as init
from brevitas.inject import BaseInjector as Injector
from brevitas.core.bit_width import BitWidthImplType
from brevitas.core.quant import QuantType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.scaling import ScalingImplType
from brevitas_examples import bnn_pynq, imagenet_classification
import time
import torch_pruning as tp

# from torch_pruning import utils
from brevitas.export import export_qonnx
from visualize_netron import showInNetron

example_inputs = torch.randn(1, 3, 32, 32)


def disable_jit(func):
    def wrapper(*args, **kwargs):
        config.JIT_ENABLED = 0
        result = func(*args, **kwargs)
        config.JIT_ENABLED = 1
        return result

    return wrapper


@disable_jit
def prune_wrapper(model, pruning_amount, pruning_mode, run_netron, folder_name):
    onnx_path_extended = f"runs/{folder_name}/extended_model"

    export_qonnx(
        model,
        args=example_inputs.cpu(),
        export_path=f"{onnx_path_extended}.onnx",
        opset_version=13,
    )
    pruning_data = prune_all_conv_layers(
        model,
        SIMD_list=[3, 32, 32, 32, 32, 32, 32, 32, 64],
        NumColPruned=pruning_amount,
        pruning_mode=pruning_mode,
    )
    pruned_onnx_filename = (
        f"{onnx_path_extended}_{str(pruning_amount).replace('.', '_')}_{pruning_mode}"
    )
    export_qonnx(
        model,
        args=example_inputs.cpu(),
        export_path=f"{pruned_onnx_filename}.onnx",
        opset_version=13,
    )
    if run_netron:
        showInNetron(f"{onnx_path_extended}.onnx", port=8080)
        showInNetron(f"{pruned_onnx_filename}.onnx", port=8081)

    with open(f"{pruned_onnx_filename}.json", "w") as fp:
        fp.write(json.dumps(pruning_data, indent=4, ensure_ascii=False))
    config.JIT_ENABLED = 1
    return model


def conv_layer_traverse(model):
    for layer in model.conv_features:
        if isinstance(layer, QuantConv2d):
            yield layer


def prune_all_conv_layers(model, SIMD_list, NumColPruned=-1, pruning_mode="structured"):
    pruning_data = []
    for layer_idx, layer in enumerate(conv_layer_traverse(model)):
        SIMD = SIMD_list[layer_idx]
        in_channels = layer.in_channels
        try:
            assert (
                in_channels % SIMD == 0
            ), f"SIMD must divide IFM Channels. Pruning {layer} is skipped."
        except AssertionError:
            continue
        if pruning_mode == "structured":
            pruning_entities = prune_brevitas_model(
                model, layer_to_prune=layer, SIMD=SIMD, NumColPruned=NumColPruned
            )
        else:
            pruning_entities = prune_brevitas_modelSIMD(
                model, layer_to_prune=layer, SIMD_in=SIMD, NumColPruned=NumColPruned
            )
        pruning_data.append(
            {
                "pruned_layer_index": layer_idx,
                "pruning_mode": pruning_mode,
                "pruning_entities": pruning_entities,
            }
        )
    return pruning_data


def prune_brevitas_modelSIMD(model, layer_to_prune, SIMD_in=1, NumColPruned=-1) -> dict:
    SIMD_out = -1
    in_channels = layer_to_prune.in_channels
    assert in_channels % SIMD_in == 0, "SIMD must divide IFM Channels"
    if isinstance(NumColPruned, float):
        SIMD_out = int(round(SIMD_in * (1.0 - NumColPruned)))
    if SIMD_out == 0:
        SIMD_out = 1
    # channels_to_prune = math.floor(model.conv_features[conv_feature_index].in_channels * pruning_amount)
    channels_to_prune = [
        i if i % SIMD_in < SIMD_in - SIMD_out else -1 for i in range(in_channels)
    ]
    channels_to_prune = list(filter(lambda x: x > -1, channels_to_prune))
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
    channels_to_prune = [i for i in range(prune_block_len)]
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


example_map = {
    ("CNV", 1, 1): bnn_pynq.cnv_1w1a,
    ("CNV", 1, 2): bnn_pynq.cnv_1w2a,
    ("CNV", 2, 2): bnn_pynq.cnv_2w2a,
    ("LFC", 1, 1): bnn_pynq.lfc_1w1a,
    ("LFC", 1, 2): bnn_pynq.lfc_1w2a,
    ("SFC", 1, 1): bnn_pynq.sfc_1w1a,
    ("SFC", 1, 2): bnn_pynq.sfc_1w2a,
    ("SFC", 2, 2): bnn_pynq.sfc_2w2a,
    ("TFC", 1, 1): bnn_pynq.tfc_1w1a,
    ("TFC", 1, 2): bnn_pynq.tfc_1w2a,
    ("TFC", 2, 2): bnn_pynq.tfc_2w2a,
    ("mobilenet", 4, 4): imagenet_classification.quant_mobilenet_v1_4b,
}


def get_test_model(netname, wbits, abits, pretrained):
    """Returns the model specified by input arguments from the Brevitas BNN-PYNQ
    test networks. Pretrained weights loaded if pretrained is True."""
    model_cfg = (netname, wbits, abits)
    model_def_fxn = example_map[model_cfg]
    fc = model_def_fxn(pretrained)
    return fc.eval()


def get_test_model_trained(netname, wbits, abits):
    "get_test_model with pretrained=True"
    return get_test_model(netname, wbits, abits, pretrained=True)


class CommonQuant(Injector):
    bit_width_impl_type = BitWidthImplType.CONST
    scaling_impl_type = ScalingImplType.CONST
    restrict_scaling_type = RestrictValueType.FP
    scaling_per_output_channel = False
    narrow_range = True
    signed = True

    @value
    def quant_type(bit_width):
        if bit_width is None:
            return QuantType.FP
        elif bit_width == 1:
            return QuantType.BINARY
        else:
            return QuantType.INT


class CommonWeightQuant(CommonQuant):
    scaling_const = 1.0


class CommonActQuant(CommonQuant):
    min_val = -1.0
    max_val = 1.0


# from .tensor_norm import TensorNorm
# from .common import CommonWeightQuant, CommonActQuant

CNV_OUT_CH_POOL = [
    (64, False),
    (64, True),
    (128, False),
    (128, True),
    (256, False),
    (256, False),
]
INTERMEDIATE_FC_FEATURES = [(256, 512), (512, 512)]
LAST_FC_IN_FEATURES = 512
LAST_FC_PER_OUT_CH_SCALING = False
POOL_SIZE = 2
KERNEL_SIZE = 3


class TensorNorm(nn.Module):
    def __init__(self, eps=1e-4, momentum=0.1):
        super().__init__()

        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.rand(1))
        self.bias = nn.Parameter(torch.rand(1))
        self.register_buffer("running_mean", torch.zeros(1))
        self.register_buffer("running_var", torch.ones(1))
        self.reset_running_stats()

    def reset_running_stats(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x):
        if self.training:
            mean = x.mean()
            unbias_var = x.var(unbiased=True)
            biased_var = x.var(unbiased=False)
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean.detach()
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * unbias_var.detach()
            inv_std = 1 / (biased_var + self.eps).pow(0.5)
            return (x - mean) * inv_std * self.weight + self.bias
        else:
            return (
                (x - self.running_mean) / (self.running_var + self.eps).pow(0.5)
            ) * self.weight + self.bias


class CNV(Module):
    def __init__(
        self, num_classes, weight_bit_width, act_bit_width, in_bit_width, in_ch
    ):
        super(CNV, self).__init__()

        self.conv_features = ModuleList()
        self.linear_features = ModuleList()

        self.conv_features.append(
            QuantIdentity(  # for Q1.7 input format
                act_quant=CommonActQuant,
                bit_width=in_bit_width,
                min_val=-1.0,
                max_val=1.0 - 2.0 ** (-7),
                narrow_range=False,
                restrict_scaling_type=RestrictValueType.POWER_OF_TWO,
            )
        )

        for out_ch, is_pool_enabled in CNV_OUT_CH_POOL:
            self.conv_features.append(
                QuantConv2d(
                    kernel_size=KERNEL_SIZE,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width,
                )
            )
            in_ch = out_ch
            self.conv_features.append(BatchNorm2d(in_ch, eps=1e-4))
            self.conv_features.append(
                QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width)
            )
            if is_pool_enabled:
                self.conv_features.append(MaxPool2d(kernel_size=2))

        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(
                QuantLinear(
                    in_features=in_features,
                    out_features=out_features,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width,
                )
            )
            self.linear_features.append(BatchNorm1d(out_features, eps=1e-4))
            self.linear_features.append(
                QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width)
            )

        self.linear_features.append(
            QuantLinear(
                in_features=LAST_FC_IN_FEATURES,
                out_features=num_classes,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width,
            )
        )
        self.linear_features.append(TensorNorm())

        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.conv_features:
            if isinstance(mod, QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.linear_features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.conv_features:
            x = mod(x)
        x = x.view(x.shape[0], -1)
        for mod in self.linear_features:
            x = mod(x)
        return x


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
    file.write(text)
    file.flush()
