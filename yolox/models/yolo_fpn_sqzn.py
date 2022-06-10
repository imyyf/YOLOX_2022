#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .squeezenet import SqueezeNet_cw_net
from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, SPPBottleneck


class SQFPN(nn.Module):
    """
    backbone is squeezenet
    """

    def __init__(
        self,
        width = 0.5,
        depth = 0.33,
        # in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="relu",
    ):
        super().__init__()
        self.stem = Focus(3, 16, ksize=3, act=act)
        self.backbone = SqueezeNet_cw_net(used_layers=[3,6]) # 147 47
        self.dark5 = nn.Sequential(
            Conv(256, 512, 3, 2, act=act),
            SPPBottleneck(512, 512, activation=act),
            CSPLayer(
                512,
                512,
                n=1,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )
        # out 1
        self.out1_cbl = self._make_cbl(512, 256, 1)
        self.out1 = self._make_embedding([256, 512], 512 + 256)

        # out 2
        self.out2_cbl = self._make_cbl(256, 128, 1)
        self.out2 = self._make_embedding([128, 256], 256 + 128)

        # upsample
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def _make_cbl(self, _in, _out, ks):
        return BaseConv(_in, _out, ks, stride=1, act="relu")

    def _make_embedding(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                self._make_cbl(in_filters, filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
                self._make_cbl(filters_list[0], filters_list[1], 3),
                self._make_cbl(filters_list[1], filters_list[0], 1),
            ]
        )
        return m

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        inputf = self.stem(input)
        features = self.backbone(inputf)
        
        [x2, x1] = features # 4,7

        x0 = self.dark5(x1)

        #  yolo branch 1
        x1_in = self.out1_cbl(x0)
        x1_in = self.upsample(x1_in)
        x1_in = torch.cat([x1_in, x1], 1)
        out_dark4 = self.out1(x1_in)

        #  yolo branch 2
        x2_in = self.out2_cbl(out_dark4)
        x2_in = self.upsample(x2_in)
        x2_in = torch.cat([x2_in, x2], 1)
        out_dark3 = self.out2(x2_in)

        outputs = (out_dark3, out_dark4, x0)

        return outputs
