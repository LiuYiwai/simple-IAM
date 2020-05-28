import torch.nn as nn


class FC_ResNet(nn.Module):

    def __init__(self, model, num_classes):
        super(FC_ResNet, self).__init__()

        # feature encoding
        self.res_block1 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool)

        self.res_block2 = model.layer1
        self.res_block3 = model.layer2
        self.res_block4 = model.layer3
        self.res_block5 = model.layer4

        # classifier
        self.num_features = model.layer4[1].conv1.in_channels
        self.classifier = nn.Sequential(
            nn.Conv2d(self.num_features, num_classes, kernel_size=1, bias=True))

    def forward(self, x):

        p1 = self.res_block1(x)
        p2 = self.res_block2(p1)
        p3 = self.res_block3(p2)
        p4 = self.res_block4(p3)
        p5 = self.res_block5(p4)

        p6 = self.classifier(p5)
        return p6, p2, p3, p4
