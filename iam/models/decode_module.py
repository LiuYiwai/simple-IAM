import torch.nn as nn
import torch.nn.functional as F


class DecodeModule(nn.Module):

    def __init__(self, channel_num=16, pool_multiple=None):
        super(DecodeModule, self).__init__()

        if pool_multiple is None:
            pool_multiple = [2, 2]
        self.pool_multiple = pool_multiple

        # decode modules
        self.decode1 = nn.Sequential(
            nn.ConvTranspose2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_num, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        self.decode2 = nn.Sequential(
            nn.ConvTranspose2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_num, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        self.decode3 = nn.Sequential(
            nn.ConvTranspose2d(channel_num, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.decode1(x)

        upsampling_size = [x.size(2), x.size(3)]
        upsampling_size = [item * self.pool_multiple[-1] for item in upsampling_size]
        x = F.interpolate(x, size=upsampling_size, mode='bilinear', align_corners=True)
        x = self.decode2(x)

        upsampling_size = [x.size(2), x.size(3)]
        upsampling_size = [item * self.pool_multiple[-2] for item in upsampling_size]
        x = F.interpolate(x, size=upsampling_size, mode='bilinear', align_corners=True)
        x = self.decode3(x)
        return x
