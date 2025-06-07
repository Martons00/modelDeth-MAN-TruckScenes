class LightSEBlock(nn.Module):
    def __init__(self, channel, reduction=8):  # Ridotto reduction
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class EfficientDepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                 padding='same', groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)  # Un solo BN

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return F.relu(self.bn(x))

class SimplifiedMultiScale(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels//2, 1)
        self.branch2 = nn.Conv2d(in_channels, out_channels//2, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        return F.relu(self.bn(torch.cat([b1, b2], dim=1)))

class Light_ScaleBiasHead(nn.Module):
    def __init__(self):
        super().__init__()
        # Fase iniziale più compatta
        self.conv1 = nn.Conv2d(4, 8, 3, padding=1)  # Canali ridotti
        self.msff = SimplifiedMultiScale(8, 16)     # Output ridotto
        self.se = LightSEBlock(16)
        self.conv3 = EfficientDepthwiseConv(16, 32) # Canali ottimizzati

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(32, 3),  # Rimossa layer intermedio
            nn.Dropout(0.2)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.msff(x)
        x = self.se(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                 stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.depthwise(x)))
        x = F.relu(self.bn2(self.pointwise(x)))
        return x
class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv2d(in_channels, out_channels//4, 1)
        self.branch2 = nn.Conv2d(in_channels, out_channels//4, 3, padding=1)
        self.branch3 = nn.Conv2d(in_channels, out_channels//4, 5, padding=2)
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels//4, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1, b2, b3, b4], dim=1)
        return F.relu(self.bn(out))

class Heavy_ScaleBiasHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, 3, padding=1)
        self.msff = MultiScaleFeatureFusion(16, 32)
        self.se = SEBlock(32)
        self.conv3 = DepthwiseSeparableConv(32, 64)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = self.msff(x)
        x = self.se(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)
    
class Moderate_ScaleBiasHead(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: [batch_size, 4, H, W] (RGB + depth)

        # Blocco convoluzionale espanso
        self.conv_block = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, padding=2),  # Più filtri
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Meno aggressivo

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        # Adaptive pooling dinamico
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Testa FC migliorata
        self.fc = nn.Sequential(
            nn.Linear(32 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3)
        )

    def forward(self, x):
        x = self.conv_block(x)        # [B,32,H/4,W/4]
        x = self.global_pool(x)       # [B,32,4,4]
        x = x.view(x.size(0), -1)     # [B, 512]
        x = self.fc(x)                # [B,3]
        return x
