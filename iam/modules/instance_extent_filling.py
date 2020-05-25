from functools import reduce

import numpy as np
import torch
import torch.nn as nn

from iam.models.decode_module import DecodeModule
from iam.models.encode_module import EncodeModule
from iam.models.feature_pyramid import FeaturePyramid


class InstanceExtentFilling(nn.Sequential):

    def __init__(self, channel_num=16, kernel=3, iterate_num=10,
                 pool_multiple=None, image_size=448, feature_map_channel=2048):
        super(InstanceExtentFilling, self).__init__()

        if pool_multiple is None:
            pool_multiple = [2, 2]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.iterate_num = iterate_num
        self.channel_num = channel_num
        self.pool_multiple = pool_multiple
        self.kernel = kernel
        self.W = self.H = image_size // reduce(lambda x, y: x * y, self.pool_multiple)
        self.norm_sum = 1

        self.encode = EncodeModule(self.channel_num, self.pool_multiple)
        self.decode = DecodeModule(self.channel_num, self.pool_multiple)
        self.feature_pyramid = FeaturePyramid(self.channel_num, self.kernel, feature_map_channel,
                                              self.H, self.W)

        self.offset = (self.kernel - 1) // 2
        mat_x_offset, mat_y_offset = np.meshgrid(np.arange(0, 2 * self.offset + 1),
                                                 np.arange(0, 2 * self.offset + 1))
        mat_x_offset = mat_x_offset.reshape(-1)
        mat_y_offset = mat_y_offset.reshape(-1)
        self.mat_x_offset = list(mat_x_offset)
        self.mat_y_offset = list(mat_y_offset)
        self.mat_x_offset_end = list(mat_x_offset + self.W)
        self.mat_y_offset_end = list(mat_y_offset + self.H)
        self.padding = torch.nn.ConstantPad2d(self.offset, 0)

    def _iterate_filling(self, inp, peak_list, weight):

        batch_num = weight.size(0)
        result = torch.zeros_like(inp)
        padding_inp = self.padding(inp)

        for batch_idx in range(batch_num):
            mask = peak_list[:, 0] == batch_idx
            if mask.sum().item() == 0:
                continue
            for i in range(self.kernel * self.kernel):
                result[mask, :, :, :] += padding_inp[
                                         mask, :,
                                         self.mat_x_offset[i]:self.mat_x_offset_end[i],
                                         self.mat_y_offset[i]:self.mat_y_offset_end[i]] * \
                                         weight[batch_idx, :, :, :, self.mat_x_offset[i], self.mat_y_offset[i]]

        return result

    def _norm_weight(self, weight):

        weight_size = weight.size()
        weight_kernel = weight.view(weight.size(0), weight.size(1),
                                    weight.size(2), weight.size(3), -1)
        weight_kernel_sum = weight_kernel.sum(4)
        weight_kernel_sum = weight_kernel_sum.unsqueeze(4)
        mask = weight_kernel_sum[:, :, :, :] == 0
        weight_kernel_sum[mask] = 1

        norm_weight = weight_kernel / weight_kernel_sum * self.norm_sum
        norm_weight = norm_weight.view(weight_size)

        return norm_weight

    def forward(self, peak_response_maps, peak_list, feature_maps):

        pyramid = self.feature_pyramid(feature_maps)
        weight = pyramid.permute(0, 2, 3, 1)
        weight = weight.view(feature_maps.shape[0], self.H, self.W,
                             self.channel_num, self.kernel, self.kernel)
        weight = weight.permute(0, 3, 1, 2, 4, 5)
        norm_weight = self._norm_weight(weight)

        inp = peak_response_maps.unsqueeze(1)
        x = self.encode(inp)

        for i in range(self.iterate_num):
            x = self._iterate_filling(x, peak_list, norm_weight)

        decode_out = self.decode(x)
        decode_out = decode_out.squeeze(1)

        return decode_out
