"""Model definitions extracted from the original notebook."""

from . import utils

# ---- cell 1 ----
import torch
import torch.nn as nn

# Depthwise Separable Convolution
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise Convolution
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        # Pointwise Convolution (채널 수 늘리기)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.leakyrelu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        return x


# Inverted Residual Block
class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion_factor):
        super(InvertedResidualBlock, self).__init__()
        self.stride = stride
        self.expansion_factor = expansion_factor

        # First pointwise convolution (expansion)
        self.expand = nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels * expansion_factor)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

        # Depthwise separable convolution
        self.depthwise = DepthwiseSeparableConv(in_channels * expansion_factor, out_channels, stride)

        # Skip connection, if stride == 1 and input/output channels are the same
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

    def forward(self, x):
        x_ = self.expand(x)
        x_ = self.bn1(x_)
        x_ = self.leakyrelu(x_)
        x_ = self.depthwise(x_)

        if self.use_res_connect:
            return x + x_
        else:
            return x_


# MobileNetV2
class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10):  # width_mult 기본값 1.0 (변경하지 않음)
        super(MobileNetV2, self).__init__()
        input_channels = 16  # 첫 번째 Conv 레이어에서의 입력 채널 수 (줄임)
        last_channels = 1280  # 마지막 레이어의 출력 채널 수

        # 첫 번째 Conv 레이어
        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        )]

        # Inverted Residual Block 설정 (출력 채널 수와 expansion factor를 늘림)
        inverted_residual_setting = [
            [8, 24, 2, 2],       # expansion_factor를 8로 변경, output_channels는 24
            [8, 48, 3, 2],       # expansion_factor를 8로 변경
            [8, 96, 3, 2],       # expansion_factor를 8로 변경
            [8, 160, 3, 2],      # expansion_factor를 8로 변경
            [8, 192, 3, 1],      # expansion_factor를 8로 변경
            [8, 320, 1, 1],     # expansion_factor를 10으로 설정
        ]

        for t, c, n, s in inverted_residual_setting:
            out_channels = c  # width_mult 없이 직접 설정된 출력 채널 수
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidualBlock(input_channels, out_channels, stride, t))
                input_channels = out_channels  # 이전 레이어의 출력 채널을 다음 레이어의 입력 채널로 설정

        # 최종 Conv 레이어와 Adaptive Pooling
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channels, last_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(last_channels),
            nn.LeakyReLU(negative_slope=0.01, inplace=True)
        ))

        self.features = nn.Sequential(*self.features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
    nn.Dropout(0.2),  # Dropout rate 30%
    nn.Linear(last_channels, num_classes)
)


    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ---- cell 2 ----
criterion = nn.CrossEntropyLoss()

# 옵티마이저 설정 (AdamW)
optimizer_ft = torch.optim.AdamW(model_ft.parameters(), lr=0.0005, weight_decay=1e-4)  # weight_decay 유지

# 학습률 스케줄러 설정 (CosineAnnealingLR)
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=training_epochs)

