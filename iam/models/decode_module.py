import torch.nn as nn
import torch.nn.functional as F


class DecodeModule(nn.Module):

    def __init__(self, channel_num=16, bn_momentum=0.1, track_running_stats=True):
        super(DecodeModule, self).__init__()

        intermediate_feature = 64

        # decode modules
        self.decode1 = nn.Sequential(
            nn.ConvTranspose2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_num, momentum=bn_momentum, track_running_stats=track_running_stats),
            nn.ReLU(inplace=True),
        )

        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(channel_num, intermediate_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(intermediate_feature, momentum=bn_momentum, track_running_stats=track_running_stats),
            nn.ReLU(inplace=True),
        )

        self.decode3 = nn.Sequential(
            nn.ConvTranspose2d(intermediate_feature, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, momentum=bn_momentum, track_running_stats=track_running_stats),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.decode1(x)

        upsampling_size = [x.size(2) * 2, x.size(3) * 2]
        x = F.interpolate(x, size=upsampling_size, mode='bilinear', align_corners=True)
        x = self.decode2(x)

        upsampling_size = [x.size(2) * 2, x.size(3) * 2]
        x = F.interpolate(x, size=upsampling_size, mode='bilinear', align_corners=True)
        x = self.decode3(x)

        return x
