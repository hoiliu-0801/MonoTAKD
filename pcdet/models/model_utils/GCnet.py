import torch
from torch import nn
from pcdet.models.model_utils.ASPP import ASPP

from torchvision.ops import DeformConv2d

class DConv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False):
        super(DConv, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias)
        self.conv2 = DeformConv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(x, out)
        return out

class SAM(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(SAM, self).__init__()
        self.inplanes = in_channels # 640
        self.planes = out_channels  # 128
        self.aspp = ASPP()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #　nn.Mish(inplace=True)
        self.conv = nn.Sequential(
                        nn.Conv2d(in_channels, 256, kernel_size=3,stride=1, padding=1, bias=False),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(256, out_channels, kernel_size=3,stride=1, padding=1, bias=False),
                        nn.ReLU(inplace=True))
        self.dcn = DConv(inplanes=self.inplanes, planes=self.planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.fc = nn.Sequential(
            nn.Linear(self.inplanes, self.planes, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.planes, self.planes, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.planes, self.inplanes, bias=False),
            nn.Sigmoid()
        )
    def SENet(self, x):
        b, c, _, _ = x.size()
        aspp_x = self.aspp(x)  # x = torch.Size([2, 128, 188, 140])
        y = self.avg_pool(aspp_x).view(b, c) # y = torch.Size([2, 128])
        y = self.fc(y).view(b, c, 1, 1)  # y = torch.Size([2, 128, 1, 1])
        # y.expand_as(x) : torch.Size([2, 128, 1, 1])-> torch.Size([2, 128, 188, 140])
        return x * y.expand_as(x)

    def forward(self, x):
        # [N, C, 1, 1]
        # channel_att_feat = self.SENet(x) # ([2, 128, 188, 140])
        # aspp_x = self.aspp(x)
        y = self.dcn(x) # ([2, 128, 188, 140])
        # fusion = aspp_x + y
        # fusion = self.conv(fusion)
        # x = self.aspp(channel_att_feat) # (2, 128, 188, 140)
        return y


class ContextBlock2d(nn.Module):

    def __init__(self, in_channels, out_channels, pool, fusions):
        super(ContextBlock2d, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = in_channels # 640
        self.planes = out_channels  # 128
        self.pool = pool
        self.fusions = fusions

        if 'att' in pool:
            self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            input_x = x
            # [N, C, H * W]
            input_x = input_x.view(batch, channel, height * width)
            # [N, 1, C, H * W]
            input_x = input_x.unsqueeze(1)
            # [N, 1, H, W]
            context_mask = self.conv_mask(x)
            # [N, 1, H * W]
            context_mask = context_mask.view(batch, 1, height * width)
            # [N, 1, H * W]
            context_mask = self.softmax(context_mask)
            # [N, 1, H * W, 1]
            context_mask = context_mask.unsqueeze(3)
            # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)
            # [N, C, 1, 1]
            context = context.view(batch, channel, 1, 1)
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x) # ([2, 128, 188, 140])->([2, 128, 1, 1])

        if self.channel_mul_conv is not None:
            # [N, C, 1, 1]
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = x * channel_mul_term
        else:
            out = x
        if self.channel_add_conv is not None:
            # [N, C, 1, 1]
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out