import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from iam.models.decode_module import DecodeModule
from iam.models.encode_module import EncodeModule
from iam.models.feature_pyramid import FeaturePyramid


class InstanceExtentFilling(nn.Sequential):

    def __init__(self, channel_num=16, kernel=3, use_checkpoints=True, iterate_num=32, sub_iterate_num=32,
                 image_size=448):
        super(InstanceExtentFilling, self).__init__()

        self.inference = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.channel_num = channel_num
        self.bn_momentum = 0.1
        self.pool_multiple = 2 * 2
        self.kernel = kernel
        self.offset = kernel // 2
        self.W = self.H = image_size // self.pool_multiple
        self.norm_sum = 1
        self.counter = None
        self.batch_list = None

        self.iterate_num = iterate_num
        self.sub_iterate_num = sub_iterate_num
        repeat_num = self.iterate_num // self.sub_iterate_num
        rest_num = self.iterate_num % self.sub_iterate_num
        self.iterate_list = [self.sub_iterate_num] * repeat_num
        self.iterate_list.append(rest_num)

        track_running_stats = True
        self.encode = EncodeModule(self.channel_num, self.bn_momentum, track_running_stats)
        self.decode = DecodeModule(self.channel_num, self.bn_momentum, track_running_stats)
        self.feature_pyramid = FeaturePyramid(self.channel_num, self.kernel, self.H, self.W,
                                              self.bn_momentum, track_running_stats)

        self.use_checkpoints = use_checkpoints

    def eval(self):
        super(InstanceExtentFilling, self).eval()
        self.inference = True
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.train()
                layer.momentum = 0

    def train(self, mode=True):
        super(InstanceExtentFilling, self).train(mode)
        self.inference = False
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.momentum = self.bn_momentum

    def _norm_weight(self, weight):

        weight_size = weight.size()
        weight_kernel = weight.view(weight.size(0), weight.size(1),
                                    weight.size(2), weight.size(3), -1)
        weight_kernel_sum = weight_kernel.sum(4, keepdim=True)
        mask = weight_kernel_sum[:, :, :, :] == 0
        weight_kernel_sum[mask] = 1

        norm_weight = weight_kernel / weight_kernel_sum * self.norm_sum
        norm_weight = norm_weight.view(weight_size)

        return norm_weight

    def _iterate_filling(self, inp, peak_list, weight):

        result = torch.zeros_like(inp)

        unfold_inp = F.unfold(inp, kernel_size=self.kernel, padding=self.offset)
        for iter_idx, batch_idx in enumerate(self.batch_list):
            mask = peak_list[:, 0] == batch_idx
            if mask.sum().item() == 0:
                continue
            temp = unfold_inp[mask].mul(weight[iter_idx])
            temp = temp.view(temp.shape[0], self.channel_num, self.kernel, self.kernel, self.H, self.W)
            temp = temp.permute(0, 1, 4, 5, 2, 3).view(temp.shape[0], self.channel_num, self.H, self.W, -1)
            result[mask] = temp.sum(4)

        return result

    def _iterate_filling_times(self, x, peak_list, weight, times):
        for i in range(int(times.item())):
            x = self._iterate_filling(x, peak_list, weight)
        return x

    def forward(self, peak_response_maps, peak_list, p2, p3, p4):

        self.counter = collections.Counter(peak_list[:, 0].cpu().numpy().tolist())
        self.batch_list = list(dict(self.counter).keys())

        pyramid = self.feature_pyramid(p2, p3, p4)

        weight = pyramid.permute(0, 2, 3, 1)
        weight = weight.view(len(self.batch_list), self.H, self.W,
                             self.channel_num, self.kernel, self.kernel)
        weight = self._norm_weight(weight)
        weight = weight.view(len(self.batch_list), self.H, self.W, -1)
        weight = weight.permute(0, 3, 1, 2).view(len(self.batch_list), weight.shape[3], -1)

        inp = peak_response_maps.unsqueeze(1)
        x = self.encode(inp)

        if self.inference == False and self.use_checkpoints:
            for it_num in self.iterate_list:
                x = cp.checkpoint(self._iterate_filling_times, x, peak_list, weight, torch.Tensor([it_num]))
        else:
            for it_num in self.iterate_list:
                x = self._iterate_filling_times(x, peak_list, weight, torch.Tensor([it_num]))

        decode_out = self.decode(x)
        decode_out = decode_out.squeeze(1)

        return decode_out
