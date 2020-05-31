from fnmatch import fnmatch
from typing import Union, List, Dict, Iterable

import torch.nn as nn
from torchvision import models

from iam.models import FC_ResNet
from iam.modules import PeakResponseMapping, InstanceExtentFilling


def finetune(
        model: nn.Module,
        base_lr: float,
        groups: Dict[str, float],
        ignore_the_rest: bool = False,
        raw_query: bool = False) -> List[Dict[str, Union[float, Iterable]]]:
    """Fintune.
    """

    # print('finetune------->> ', base_lr, groups, ignore_the_rest, raw_query)

    parameters = [dict(params=[], names=[], query=query if raw_query else '*' + query + '*', lr=lr * base_lr) for
                  query, lr in groups.items()]
    rest_parameters = dict(params=[], names=[], lr=base_lr)
    for k, v in model.named_parameters():
        for group in parameters:
            if fnmatch(k, group['query']):
                group['params'].append(v)
                group['names'].append(k)
            else:
                rest_parameters['params'].append(v)
                rest_parameters['names'].append(k)
    if not ignore_the_rest:
        parameters.append(rest_parameters)
    for group in parameters:
        group['params'] = iter(group['params'])
    return parameters


def fc_resnet50(num_classes: int = 20, pretrained: bool = True) -> nn.Module:
    """FC ResNet50.
    """
    model = FC_ResNet(models.resnet50(pretrained), num_classes)
    return model


def peak_response_mapping(
        backbone: nn.Module,
        enable_peak_stimulation: bool = True,
        enable_peak_backprop: bool = True,
        win_size: int = 3,
        sub_pixel_locating_factor: int = 1,
        filter_type: Union[str, int, float] = 'median') -> nn.Module:
    """Peak Response Mapping.
    """

    model = PeakResponseMapping(
        backbone,
        enable_peak_stimulation=enable_peak_stimulation,
        enable_peak_backprop=enable_peak_backprop,
        win_size=win_size,
        sub_pixel_locating_factor=sub_pixel_locating_factor,
        filter_type=filter_type)
    return model


def instance_extent_filling(config):
    model = InstanceExtentFilling(channel_num=config['encode_channel'],
                                  kernel=config['filling_kernel'],
                                  use_checkpoints=config['enable_filling_forward_checkpoints'],
                                  iterate_num=config['iterate_num'],
                                  sub_iterate_num=config['sub_iterate_num'],
                                  image_size=config['image_size'],
                                  )
    return model
