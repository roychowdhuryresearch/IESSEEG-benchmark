import torch
import torch.nn as nn
import torch.nn.functional as F

def conv3x3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """
    3D convolution with padding
    """
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation
    )

def conv1x1x1(in_planes, out_planes, stride=1):
    """
    1x1x1 convolution
    """
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock3D(nn.Module):
    """
    A 3D version of the basic residual block used in ResNet-18/34.
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(BasicBlock3D, self).__init__()
        # Both self.conv1 and self.conv2 have stride=1 except the first block of each layer,
        # which may have stride > 1 (if specified).
        self.conv1 = conv3x3x3(inplanes, planes, stride, groups, dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, groups=groups, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck3D(nn.Module):
    """
    A 3D version of the bottleneck block used in ResNet-50/101/152.
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1):
        super(Bottleneck3D, self).__init__()
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1x1(inplanes, width)
        self.bn1 = nn.BatchNorm3d(width)
        self.conv2 = conv3x3x3(width, width, stride, groups, dilation)
        self.bn2 = nn.BatchNorm3d(width)
        self.conv3 = conv1x1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    """
    Generic 3D ResNet. 
    Pass BasicBlock3D for layers=[2,2,2,2] => ResNet18-like.
    Pass Bottleneck3D for layers=[3,4,6,3] => ResNet50-like.
    """
    def __init__(self,
                 block,
                 layers,
                 in_channels=1,
                 num_classes=2,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 dropout_prob=0.0):
        super(ResNet3D, self).__init__()
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self._norm_layer = nn.BatchNorm3d

        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.dropout = nn.Dropout(p=dropout_prob)

        # Initial conv layer => out_planes=64
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=7, stride=(2,2,2),
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Build layers
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # Final classification head
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Optionally initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block
        # behaves like an identity. (Improves training stability.)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck3D):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock3D):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """
        planes: output dimension for this layer
        blocks: how many blocks in this layer
        stride: stride for the first block
        dilate: if True, use dilated conv
        """
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        # If we need to match dimension or stride
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                self._norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x => (batch_size, in_channels, depth, height, width)
        x = self.conv1(x)   # => shape: (batch_size, 64, D/2, H/2, W/2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # => shape: (batch_size, 64, D/4, H/4, W/4)

        x = self.layer1(x)  # => shape: (batch_size, 64, ...)
        x = self.layer2(x)  # => shape: (batch_size, 128, ...)
        x = self.layer3(x)  # => shape: (batch_size, 256, ...)
        x = self.layer4(x)  # => shape: (batch_size, 512, ...)

        x = self.avgpool(x) # => shape: (batch_size, 512, 1,1,1)
        x = torch.flatten(x, 1)  # => (batch_size, 512)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ------------------------------------------------------------
# Helper constructors for common 3D-ResNet variants
# ------------------------------------------------------------
def resnet18_3d(in_channels=1, num_classes=2):
    """
    Build a 3D ResNet-18 model.
    """
    return ResNet3D(
        block=BasicBlock3D,
        layers=[2, 2, 2, 2],
        in_channels=in_channels,
        num_classes=num_classes
    )

def resnet50_3d(in_channels=1, num_classes=2):
    """
    Build a 3D ResNet-50 model.
    """
    return ResNet3D(
        block=Bottleneck3D,
        layers=[3, 4, 6, 3],
        in_channels=in_channels,
        num_classes=num_classes
    )
