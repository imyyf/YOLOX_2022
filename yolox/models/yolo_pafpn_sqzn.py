#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .squeezenet import SqueezeNet_cw_net
from .network_blocks import BaseConv, CSPLayer, DWConv


class SQPAFPN(nn.Module):
    """
    backbone is squeezenet
    """

    def __init__(
        self,
        # in_features=("dark3", "dark4", "dark5"),
        # in_channels=[256, 512, 1024],
        depthwise=False,
        act="relu",
    ):
        super().__init__()
        self.backbone = SqueezeNet_cw_net(used_layers=[4,7])
        # self.in_features = in_features
        # self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            256, 128, 1, 1, act=act
        )

        self.pan1 = CSPLayer(
            256,
            128,
            1,
            False,
            depthwise=depthwise,
            act=act,
        )

        self.pan0 = CSPLayer(
            256,
            256,
            1,
            False,
            depthwise=depthwise,
            act=act,
        )

        # self.C3_p4 = CSPLayer(
        #     int(2 * in_channels[1] * width),
        #     int(in_channels[1] * width),
        #     round(3 * depth),
        #     False,
        #     depthwise=depthwise,
        #     act=act,
        # )  # cat

        
        # self.C3_p3 = CSPLayer(
        #     int(2 * in_channels[0] * width),
        #     int(in_channels[0] * width),
        #     round(3 * depth),
        #     False,
        #     depthwise=depthwise,
        #     act=act,
        # )

        # bottom-up conv
        # self.bu_conv2 = Conv(
        #     int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        # )
        # self.C3_n3 = CSPLayer(
        #     int(2 * in_channels[0] * width),
        #     int(in_channels[1] * width),
        #     round(3 * depth),
        #     False,
        #     depthwise=depthwise,
        #     act=act,
        # )

        # bottom-up conv
        self.bu_conv1 = Conv(
            128, 128, 3, 2, act=act
        )
        # self.C3_n4 = CSPLayer(
        #     int(2 * in_channels[1] * width),
        #     int(in_channels[2] * width),
        #     round(3 * depth),
        #     False,
        #     depthwise=depthwise,
        #     act=act,
        # )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone
        features = self.backbone(input)
        
        [x1, x0] = features # 4,7

        fpn_out0 = self.lateral_conv0(x0)
        f_out0 = self.upsample(fpn_out0)
        f_out0 = torch.cat([f_out0, x1], 1)
        pan_out1 = self.pan1(f_out0)

        p_out0 = self.bu_conv1(pan_out1)
        p_out0 = torch.cat([p_out0, fpn_out0], 1)
        pan_out0 = self.pan0(p_out0)
        # fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        # f_out0 = self.upsample(fpn_out0)  # 512/16
        # f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        # f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        # fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        # f_out1 = self.upsample(fpn_out1)  # 256/8
        # f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        # pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        # p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        # p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        # pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        # p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        # p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        # pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        outputs = (pan_out1, pan_out0)
        return outputs
