import torch
import json
import brevitas.config as config
from brevitas.nn import QuantConv2d
import numpy as np

# from brevitas.core.restrict_val import RestrictValueType
import time
import torch_pruning as tp
import torch.nn as nn
import matplotlib.pyplot as plt

# from torch_pruning import utils
from brevitas.export import export_qonnx
from visualize_netron import showInNetron
from configurations import (
    now_time,
    pruning_amount,
    cmd_args,
    SqrHingeLoss,
    weight_decay,
    lr,
    logger,
    lr_schedule_period,
    lr_schedule_ratio,
)
from models_folder.models import model_with_cfg, extended_model_with_cfg

example_inputs = torch.randn(1, 3, 32, 32)
SIMD_LIST = [3, 32, 32, 32, 32, 32, 32, 32, 64]
PE_LIST = [16, 32, 16, 16, 4, 1, 1, 1, 5]
keep_weights_intact = True

run_netron, pruning_mode, use_scheduler, model_identity, is_iterative, pretrained, is_extended = (
    cmd_args["run_netron"],
    cmd_args["pruning_mode"],
    cmd_args["use_scheduler"],
    cmd_args["model_identity"],
    cmd_args["is_iterative"],
    cmd_args["pretrained"],
    cmd_args["is_extended"],
)


class MaskedLayer(nn.Module):
    def __init__(self, mask):
        super().__init__()
        self.channel_to_prune = mask
        self.first_run = True

    def forward(self, x):
        if self.first_run:
            mask_inp = torch.ones_like(x)
            mask_inp[:, self.channel_to_prune, :, :] = 0
            self.register_buffer('mask', mask_inp)
            self.first_run = False
        # x shape: (batch_size, in_channels, H, W)
        masked_x = torch.mul(x, self.mask)
        return masked_x


def disable_jit(func):
    def wrapper(*args, **kwargs):
        config.JIT_ENABLED = 0
        result = func(*args, **kwargs)
        config.JIT_ENABLED = 1
        return result

    return wrapper


def weight_to_im2col(tensor):
    return tensor.reshape(tensor.shape[0] * tensor.shape[1] * tensor.shape[2], -1)


def im2col_to_weight(tensor, ifm_size, kernel_size=(3, 3)):
    return tensor.reshape(ifm_size, kernel_size[0], kernel_size[1], -1).permute(3, 0, 1, 2)


class OneShotPruning():
    def __init__(self, is_extended):
        self.total_num_of_pruned_layers = 0
        if is_extended:
            self.updated_model, _ = extended_model_with_cfg(model_identity, False)
        else:
            self.updated_model, _ = model_with_cfg(model_identity, pretrained=pretrained)
    @disable_jit
    def prune_wrapper(self, model, pruning_amount, pruning_mode, run_netron, folder_name):
        onnx_path_extended = f"{folder_name}/extended_model"

        export_qonnx(
            model,
            args=example_inputs.cpu(),
            export_path=f"{onnx_path_extended}.onnx",
            opset_version=13,
        )
        pruning_data = self.prune_all_conv_layers(
            model,
            SIMD_list=SIMD_LIST,
            NumColPruned=pruning_amount,
            pruning_mode=pruning_mode,
        )
        self.updated_model.register_masks()
        pruned_onnx_filename = f"{onnx_path_extended}_pruned"
        # export_qonnx(
        #     self.updated_model,
        #     args=example_inputs.cpu(),
        #     export_path=f"{pruned_onnx_filename}.onnx",
        #     opset_version=13,
        # )
        if run_netron:
            showInNetron(f"{onnx_path_extended}.onnx", port=8080)
            showInNetron(f"{pruned_onnx_filename}.onnx", port=8082)
        with open(f"{pruned_onnx_filename}.json", "w") as fp:
            fp.write(json.dumps(pruning_data, indent=4, ensure_ascii=False))
        config.JIT_ENABLED = 1
        return self.updated_model


    def conv_layer_traverse(self, model):
        for position, layer in enumerate(model.conv_features):
            if isinstance(layer, QuantConv2d):
                yield position, layer


    def weight_histograms(self, model, folder_name):
        length = sum(1 for _ in self.conv_layer_traverse(model))
        figure_size = (24, 15)
        fig, axs = plt.subplots(length, figsize=figure_size, constrained_layout=True)
        for layer_idx, (position, layer) in enumerate(self.conv_layer_traverse(model)):
            tensor = self.get_layer_tensor(layer.weight.data)
            p50 = np.percentile(tensor, 50)
            p70 = np.percentile(tensor, 70)
            p85 = np.percentile(tensor, 85)
            axs[layer_idx].hist(tensor, bins=20, edgecolor="black", alpha=0.7)
            axs[layer_idx].axvline(
                p50,
                color="red",
                linestyle="dashed",
                linewidth=3,
                label="50th Percentile (Median)",
            )
            axs[layer_idx].axvline(
                p70, color="green", linestyle="dashed", linewidth=3, label="70th Percentile"
            )
            axs[layer_idx].axvline(
                p85, color="blue", linestyle="dashed", linewidth=3, label="85th Percentile"
            )
            if layer_idx == 0:
                plt.figlegend()
        plt.savefig(f"./{folder_name}/weight_hist.png")


    def fix_pe_parameter(self, model, list_of_dicts, pruning_mode):
        for layer_idx, (position, layer) in enumerate(self.conv_layer_traverse(model)):
            cur_layer_dict = [i for i in list_of_dicts if i["pruned_layer_index"] == layer_idx]
            if not len(cur_layer_dict) == 1:
                continue
            cur_layer_dict[0]["pruning_entities"]["out_channels_new"] = layer.out_channels

        for dict_i, layer_dict in enumerate(list_of_dicts):
            layer_idx = layer_dict["pruned_layer_index"]
            prev_layer = [(index, i) for index, i in enumerate(list_of_dicts) if i["pruned_layer_index"] == layer_idx - 1]
            if not len(prev_layer) == 1:
               continue
            prev_idx_in_dict, previous_layer = prev_layer[0]
            pe_previous = PE_LIST[previous_layer["pruned_layer_index"]]
            pruning_factor = float(previous_layer["pruning_entities"]["out_channels_old"] / previous_layer["pruning_entities"]["out_channels_new"])
            channel_to_pe = int(previous_layer["pruning_entities"]["out_channels_old"] / pe_previous)
            pe_new = pe_previous
            try:
                assert previous_layer["pruning_entities"]["out_channels_new"] % pe_previous == 0, f"PE should divide OFM for {previous_layer['pruned_layer_index']}"
                if pruning_mode == "SIMD" and pruning_factor.is_integer():
                    pe_new = pe_previous / pruning_factor
            except AssertionError:
                if previous_layer["pruning_entities"]["out_channels_new"] % channel_to_pe == 0:
                    pe_new = previous_layer["pruning_entities"]["out_channels_new"] / channel_to_pe
                else:
                    pe_new = 1
                    logger.warn("PE is set 1")
            list_of_dicts[prev_idx_in_dict]["pruning_entities"]["PE_old"] = int(pe_previous)
            list_of_dicts[prev_idx_in_dict]["pruning_entities"]["PE_new"] = int(pe_new)
            if dict_i == len(list_of_dicts) - 1:
                layer_dict["pruning_entities"]["PE_old"] = PE_LIST[layer_dict["pruned_layer_index"]]
                layer_dict["pruning_entities"]["PE_new"] = PE_LIST[layer_dict["pruned_layer_index"]]
        return list_of_dicts


    def prune_all_conv_layers(self, model, SIMD_list, NumColPruned=-1, pruning_mode="structured"):
        pruning_data = []
        for layer_idx, (position, layer) in enumerate(self.conv_layer_traverse(model)):
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
                pruning_entities = self.prune_brevitas_model(
                    model, layer_to_prune=layer, SIMD=SIMD, NumColPruned=pruning_ratio, position=position
                )
            else:
                pruning_entities = self.prune_brevitas_modelSIMD(
                    model, layer_to_prune=layer, SIMD_in=SIMD, NumColPruned=pruning_ratio
                )
            pruning_data.append(
                {
                    "pruned_layer_index": layer_idx,
                    "pruning_mode": pruning_mode,
                    "pruning_entities": pruning_entities,
                }
            )
        self.fix_pe_parameter(model, pruning_data, pruning_mode)
        return pruning_data


    def sort_tensor(self, tensor, SIMD=32):

        x = tensor.permute(1, 2, 3, 0)
        xx = weight_to_im2col(x)
        xx = xx.T
        chunked_by_simd = xx.chunk((x.shape[0] * x.shape[1] * x.shape[2]) // SIMD, dim=1)
        #out = xx.abs().sum(dim=1, keepdim=True)
        out = torch.tensor(list(map(lambda itm: itm.abs().sum().item(), chunked_by_simd)))
        sorted_indices = out.argsort(dim=0)
        return [i.item() for i in sorted_indices.squeeze()]
    def get_pruning_mask(self, cols_to_prune, weight_tensor):
        mask = torch.ones_like(weight_tensor)
        x = mask.permute(1, 2, 3, 0)
        xx = weight_to_im2col(x)
        xx = xx.T
        xx[:, cols_to_prune] = 0
        mask_unfolded = im2col_to_weight(xx, weight_tensor.shape[1], (3,3))
        weight_tensor[mask_unfolded == 0] *= 0
        return mask_unfolded, weight_tensor
    def get_layer_tensor(self, tensor):
        x = tensor.permute(1, 0, 2, 3)
        x = x.reshape(tensor.shape[1], -1)
        out = x.abs().mean(dim=1, keepdim=True)
        return out.squeeze()


    def prune_brevitas_modelSIMD(
        self, model, layer_to_prune, SIMD_in=1, NumColPruned=-1, SIMD_out=-1
    ) -> dict:
        in_channels = layer_to_prune.in_channels
        out_channels = layer_to_prune.out_channels
        assert in_channels % SIMD_in == 0, "SIMD must divide IFM Channels"
        num_of_blocks = int(in_channels / SIMD_in)
        if SIMD_out == -1:
            if isinstance(NumColPruned, float):
                SIMD_out = round(SIMD_in * (1.0 - NumColPruned))
            else:
                SIMD_out = (in_channels - NumColPruned) / (in_channels / SIMD_in)
            if SIMD_out == 0:
                SIMD_out = 1
        if isinstance(SIMD_out, float):
            assert SIMD_out.is_integer(), "Output SIMD should be integer"
        SIMD_out = int(SIMD_out)
        in_channels_new = num_of_blocks * SIMD_out
        # channels_to_prune = math.floor(model.conv_features[conv_feature_index].in_channels * pruning_amount)
        sorting_indices = self.sort_tensor(layer_to_prune.weight.data)
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
            layer_to_prune.weight.data[:, channels_to_prune] *= 0
            # group.prune()
        print(layer_to_prune.in_channels)
        return {
            "in_channels_old": in_channels,
            "in_channels_new": layer_to_prune.in_channels,
            "out_channels_old": out_channels,
            "SIMD_in": SIMD_in,
            "SIMD_out": SIMD_out,
        }


    def prune_brevitas_model(self, model, layer_to_prune, SIMD=1, NumColPruned=-1, position=-1) -> dict:
        in_channels = layer_to_prune.in_channels
        print(f"in_channels={in_channels} SIMD={SIMD}")
        prune_block_len = NumColPruned
        sorting_indices = self.sort_tensor(layer_to_prune.weight.data, SIMD=SIMD)
        if isinstance(NumColPruned, float):
            NumColPruned = round(len(sorting_indices)*NumColPruned)
        cols_to_prune = sorting_indices[:NumColPruned]
        mask_tensor, self.updated_model.conv_features[position].weight.data = self.get_pruning_mask(cols_to_prune, layer_to_prune.weight.data)
        self.updated_model.mask_dict[f"{position}"] = mask_tensor
        # self.updated_model.conv_features[position + self.total_num_of_pruned_layers].weight.data[:, channels_to_prune,:,:] *= 0
        #mask = torch.ones(in_channels, dtype=torch.int)
        #mask[channels_to_prune] = 0
        #mask_layer = MaskedLayer(channels_to_prune)
        ####alper sonself.updated_model.conv_features.insert(position + self.total_num_of_pruned_layers, mask_layer)
        ###alperself.total_num_of_pruned_layers += 1


        return {
            "in_channels_old": in_channels,
            "in_channels_new": layer_to_prune.in_channels,
            "out_channels_old": layer_to_prune.out_channels,
            "SIMD_in": SIMD,
            "cols_to_prune": cols_to_prune,
        }


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
