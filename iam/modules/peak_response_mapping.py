from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functions import pr_conv2d, peak_stimulation


class PeakResponseMapping(nn.Sequential):

    def __init__(self, *args, **kargs):
        super(PeakResponseMapping, self).__init__(*args)

        self.inferencing = False
        # use global average pooling to aggregate responses if peak stimulation is disabled
        self.enable_peak_stimulation = kargs.get('enable_peak_stimulation', True)
        # return only the class response maps in inference mode if peak backpropagation is disabled
        self.enable_peak_backprop = kargs.get('enable_peak_backprop', True)
        # window size for peak finding
        self.win_size = kargs.get('win_size', 3)
        # sub-pixel peak finding
        self.sub_pixel_locating_factor = kargs.get('sub_pixel_locating_factor', 1)
        # peak filtering
        self.filter_type = kargs.get('filter_type', 'median')
        if self.filter_type == 'median':
            self.peak_filter = self._median_filter
        elif self.filter_type == 'mean':
            self.peak_filter = self._mean_filter
        elif self.filter_type == 'max':
            self.peak_filter = self._max_filter
        elif isinstance(self.filter_type, (int, float)):
            self.peak_filter = lambda x: self.filter_type
        else:
            self.peak_filter = None

    @staticmethod
    def _median_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.median(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _mean_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold = torch.mean(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    @staticmethod
    def _max_filter(input):
        batch_size, num_channels, h, w = input.size()
        threshold, _ = torch.max(input.view(batch_size, num_channels, h * w), dim=2)
        return threshold.contiguous().view(batch_size, num_channels, 1, 1)

    def _patch(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module._original_forward = module.forward
                module.forward = MethodType(pr_conv2d, module)

    def _recover(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) and hasattr(module, '_original_forward'):
                module.forward = module._original_forward

    def train(self, mode=True):
        super(PeakResponseMapping, self).train(mode)
        if self.inferencing:
            self._recover()
            self.inferencing = False
        return self

    def inference(self):
        super(PeakResponseMapping, self).train(False)
        self._patch()
        self.inferencing = True
        return self

    def forward(self, _input, class_threshold=0, peak_threshold=30):

        assert _input.dim() == 4, 'PeakResponseMapping layer only supports batch mode.'

        if self.inferencing:
            _input.requires_grad_()

        # classification network forwarding
        class_response_maps, feature_maps = super(PeakResponseMapping, self).forward(_input)

        if self.enable_peak_stimulation:
            # sub-pixel peak finding
            if self.sub_pixel_locating_factor > 1:
                class_response_maps = F.interpolate(class_response_maps, scale_factor=self.sub_pixel_locating_factor,
                                                    mode='bilinear', align_corners=True)
            # aggregate responses from informative receptive fields estimated via class peak responses
            peak_list, aggregation = peak_stimulation(class_response_maps, win_size=self.win_size,
                                                      peak_filter=self.peak_filter)
        else:
            # aggregate responses from all receptive fields
            peak_list, aggregation = None, F.adaptive_avg_pool2d(class_response_maps, 1).squeeze(2).squeeze(2)

        if self.inferencing:

            if not self.enable_peak_backprop:
                # extract only class-aware visual cues
                return aggregation, class_response_maps

            # extract instance-aware visual cues, i.e., peak response maps
            assert class_response_maps.size(
                0) == 1, 'Currently inference mode (with peak backpropagation) only supports one image at a time.'
            if peak_list is None:
                peak_list = peak_stimulation(class_response_maps, return_aggregation=False, win_size=self.win_size,
                                             peak_filter=self.peak_filter)

            peak_response_maps = []
            valid_peak_list = []
            # peak backpropagation
            grad_output = class_response_maps.new_empty(class_response_maps.size())
            for idx in range(peak_list.size(0)):
                if aggregation[peak_list[idx, 0], peak_list[idx, 1]] >= class_threshold:
                    peak_val = class_response_maps[
                        peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]]
                    if peak_val > peak_threshold:
                        grad_output.zero_()
                        # starting from the peak
                        grad_output[peak_list[idx, 0], peak_list[idx, 1], peak_list[idx, 2], peak_list[idx, 3]] = 1
                        if _input.grad is not None:
                            _input.grad.zero_()
                        class_response_maps.backward(grad_output, retain_graph=True)
                        prm = _input.grad.detach().sum(1).clone().clamp(min=0)
                        prm = prm[peak_list[idx, 0]].unsqueeze(0)
                        peak_response_maps.append(prm / prm.sum())
                        valid_peak_list.append(peak_list[idx, :])

            # return results
            class_response_maps = class_response_maps.detach()
            aggregation = aggregation.detach()

            if len(peak_response_maps) > 0:
                valid_peak_list = torch.stack(valid_peak_list)
                peak_response_maps = torch.cat(peak_response_maps, 0)
                return_tuple = (aggregation, class_response_maps, valid_peak_list, peak_response_maps)
                return return_tuple, feature_maps.detach()
            else:
                return None
        else:
            # classification confidence scores
            return aggregation
