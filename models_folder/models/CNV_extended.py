# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from torch.nn import BatchNorm1d
from torch.nn import BatchNorm2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList

from brevitas.core.restrict_val import RestrictValueType
from brevitas.nn import QuantConv2d
from brevitas.nn import QuantIdentity
from brevitas.nn import QuantLinear

from .common import CommonActQuant
from .common import CommonWeightQuant
from .tensor_norm import TensorNorm

# CNV_OUT_CH_POOL = [(64, False), (64, True), (128, False), (128, True), (256, False), (256, False)]
# INTERMEDIATE_FC_FEATURES = [(256, 512), (512, 512)]
CNV_OUT_CH_POOL = [
    (64, False),
    (64, True),
    (128, False),
    (128, False),
    (256, False),
    (256, False),
    (256, False),
    (512, False),
    (512, False),
]
INTERMEDIATE_FC_FEATURES = [(512, 1024), (1024, 1024)]
LAST_FC_IN_FEATURES = 1024
LAST_FC_PER_OUT_CH_SCALING = False
POOL_SIZE = 2
KERNEL_SIZE = 3
LAST_LAYER_KERNEL_SIZE = 2


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

        for i, (out_ch, is_pool_enabled) in enumerate(CNV_OUT_CH_POOL):
            self.conv_features.append(
                QuantConv2d(
                    kernel_size=KERNEL_SIZE
                    if not i + 1 == len(CNV_OUT_CH_POOL)
                    else LAST_LAYER_KERNEL_SIZE,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    bias=False,
                    weight_quant=CommonWeightQuant,
                    weight_bit_width=weight_bit_width,
                )
            )
            # print(self.conv_features[-1])
            in_ch = out_ch
            self.conv_features.append(BatchNorm2d(in_ch, eps=1e-4))
            self.conv_features.append(
                QuantIdentity(act_quant=CommonActQuant, bit_width=act_bit_width)
            )
            if is_pool_enabled:
                self.conv_features.append(MaxPool2d(kernel_size=2))
                # print(self.conv_features[-1])

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
        for i, mod in enumerate(self.conv_features):
            # if isinstance(mod, MaxPool2d):
            #    print(mod)
            #    print(x.shape)
            # if isinstance(mod, QuantConv2d):
            #    print(mod)
            #    print(x.shape)
            try:
                x = mod(x)
            except Exception:
                pass
                # print(e)
                # print("i --->", i)
                # print(mod)
                # print(x.shape)
                # exit()
        x = x.view(x.shape[0], -1)
        for mod in self.linear_features:
            # print(x.shape)
            x = mod(x)
        # print(x.shape)
        # exit()
        return x


def cnv(cfg):
    weight_bit_width = cfg.getint("QUANT", "WEIGHT_BIT_WIDTH")
    act_bit_width = cfg.getint("QUANT", "ACT_BIT_WIDTH")
    in_bit_width = cfg.getint("QUANT", "IN_BIT_WIDTH")
    num_classes = cfg.getint("MODEL", "NUM_CLASSES")
    in_channels = cfg.getint("MODEL", "IN_CHANNELS")
    net = CNV(
        weight_bit_width=weight_bit_width,
        act_bit_width=act_bit_width,
        in_bit_width=in_bit_width,
        num_classes=num_classes,
        in_ch=in_channels,
    )
    return net
