import torch.nn as nn


class EncodeModule(nn.Module):

    def __init__(self, channel_num=16, pool_multiple=None):
        super(EncodeModule, self).__init__()

        if pool_multiple is None:
            pool_multiple = [2, 2]
        self.pool_multiple = pool_multiple

        # track_running_stats = True
        track_running_stats = False

        # encode modules
        self.encode1 = nn.Sequential(
            nn.Conv2d(1, channel_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_num, track_running_stats=track_running_stats),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=self.pool_multiple[0], stride=self.pool_multiple[0]),
        )

        self.encode2 = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_num, track_running_stats=track_running_stats),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=self.pool_multiple[1], stride=self.pool_multiple[1]),
        )

    def forward(self, x):
        x = self.encode1(x)
        x = self.encode2(x)
        return x
