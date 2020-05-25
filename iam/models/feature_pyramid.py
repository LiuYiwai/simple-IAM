import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramid(nn.Module):

    def __init__(self, channel_num=8, kernel=3, input_feature=2048, H=112, W=112):
        super(FeaturePyramid, self).__init__()

        self.H = H
        self.W = W
        self.r = kernel
        self.C = channel_num
        feature_num = self.r * self.r * self.C
        intermediate_feature = input_feature // 4

        # track_running_stats = True
        track_running_stats = False

        self.feature1 = nn.Sequential(
            nn.Conv2d(input_feature, intermediate_feature, kernel_size=1, stride=1),
            nn.BatchNorm2d(intermediate_feature, track_running_stats=track_running_stats),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False),
        )

        self.feature2 = nn.Sequential(
            nn.Conv2d(intermediate_feature, feature_num, kernel_size=1, stride=1),
            nn.BatchNorm2d(feature_num, track_running_stats=track_running_stats),
            # nn.ReLU(inplace=True),
        )

        self.lateral_layer2 = nn.Sequential(
            nn.Conv2d(intermediate_feature, feature_num, kernel_size=1, stride=1),
            nn.BatchNorm2d(feature_num, track_running_stats=track_running_stats),
        )

        self.smooth2 = nn.Sequential(
            nn.Conv2d(feature_num, feature_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_num, track_running_stats=track_running_stats),
        )

        self.smooth_out = nn.Sequential(
            nn.Conv2d(feature_num, feature_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_num, track_running_stats=track_running_stats),
            nn.ReLU(inplace=True),
        )

    def forward(self, c1):
        c2 = self.feature1(c1)
        p3 = self.feature2(c2)

        p2_size = (c2.size(2), c2.size(3))
        p3 = F.interpolate(p3, size=p2_size, mode='bilinear', align_corners=True)
        p2 = p3 + self.lateral_layer2(c2)
        p2 = self.smooth2(p2)

        p1_size = (c1.size(2), c1.size(3))
        p3 = F.interpolate(p3, size=p1_size, mode='bilinear', align_corners=True)
        p2 = F.interpolate(p2, size=p1_size, mode='bilinear', align_corners=True)

        out = p2 + p3
        out = F.interpolate(out, size=(self.H, self.W), mode='bilinear', align_corners=True)
        out = self.smooth_out(out)

        return out
