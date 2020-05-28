import torch.nn as nn
import torch.nn.functional as F


class FeaturePyramid(nn.Module):

    def __init__(self, channel_num=8, kernel=3, H=56, W=56, track_running_stats=True):
        super(FeaturePyramid, self).__init__()

        self.H = H
        self.W = W
        self.r = kernel
        self.C = channel_num
        feature_num = self.r * self.r * self.C
        intermediate_feature = 256

        self.toplayer = nn.Conv2d(1024, intermediate_feature, kernel_size=1, stride=1)

        self.latlayer3 = nn.Sequential(
            nn.Conv2d(512, intermediate_feature, kernel_size=1, stride=1),
            nn.BatchNorm2d(intermediate_feature, track_running_stats=track_running_stats),
        )

        self.latlayer2 = nn.Sequential(
            nn.Conv2d(256, intermediate_feature, kernel_size=1, stride=1),
            nn.BatchNorm2d(intermediate_feature, track_running_stats=track_running_stats),
        )

        self.smooth3 = nn.Sequential(
            nn.Conv2d(intermediate_feature, intermediate_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(intermediate_feature, track_running_stats=track_running_stats),
        )

        # out
        self.smooth2 = nn.Sequential(
            nn.Conv2d(intermediate_feature, feature_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_num, track_running_stats=track_running_stats),
        )

        self.ReLU = nn.ReLU(inplace=True)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, c2, c3, c4):
        p4 = self.toplayer(c4)
        p3 = self._upsample_add(p4, self.latlayer3(c3))
        p3 = self.smooth3(p3)
        p2 = self._upsample_add(p3, self.latlayer2(c2))
        p2 = self.smooth2(p2)
        out = self.ReLU(p2)
        return out
