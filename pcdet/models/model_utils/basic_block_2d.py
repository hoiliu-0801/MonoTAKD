import torch.nn as nn


class BasicBlock2D(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        """
        Initializes convolutional block
        Args:
            in_channels: int, Number of input channels
            out_channels: int, Number of output channels
            **kwargs: Dict, Extra arguments for nn.Conv2d
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        """
        Applies convolutional block
        Args:
            features: (B, C_in, H, W), Input features
        Returns:
            x: (B, C_out, H, W), Output features
        """
        x = self.conv(features)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicBlock2D_copy(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_channels = 128
        self.out_channels = 128
        self.conv = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        x = self.conv(features)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicBlock2D_copy2(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, features):
        x = self.conv(features)
        x = self.bn(x)
        x = self.relu(x)
        return x


# class Image_restore_net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=1, padding=1)
#         self.pool2 = nn.MaxPool2d(kernel_size=(2, 2))

#         self.fc3 = nn.Linear(8192, 512)
#         self.act3 = nn.ReLU()
#         self.drop3 = nn.Dropout(0.5)

#         self.fc4 = nn.Linear(512, 10)
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.act = nn.ReLU(inplace=True)
#         self.drop1 = nn.Dropout(0.3)
#         self.flat = nn.Flatten()
#     def forward(self, x):
#         # input 3x32x32, output 32x32x32
#         x = self.act1(self.conv1(x))
#         x = self.drop1(x)
#         # input 32x32x32, output 32x32x32
#         x = self.act2(self.conv2(x))
#         # input 32x32x32, output 32x16x16
#         x = self.pool2(x)
#         # input 32x16x16, output 8192
#         x = self.flat(x)
#         # input 8192, output 512
#         x = self.act3(self.fc3(x))
#         x = self.drop3(x)
#         # input 512, output 10
#         x = self.fc4(x)
#         return x