import torch.nn as nn


class EncodeModule(nn.Module):

    def __init__(self, channel_num=16, bn_momentum=0.1, track_running_stats=True):
        super(EncodeModule, self).__init__()

        intermediate_feature = 64

        # encode modules
        self.encode1 = nn.Sequential(
            nn.Conv2d(1, intermediate_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(intermediate_feature, momentum=bn_momentum, track_running_stats=track_running_stats),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.encode2 = nn.Sequential(
            nn.Conv2d(intermediate_feature, channel_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_num, momentum=bn_momentum, track_running_stats=track_running_stats),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        x = self.encode1(x)
        x = self.encode2(x)
        return x
