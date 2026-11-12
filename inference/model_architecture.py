import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Double convolution block used by the trained ReliefLink V5 model."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ArtifactAwareUNet(nn.Module):
    """Architecture recovered from the final V5 inference cell in the notebook."""

    def __init__(self, in_channels=3, out_channels=3, num_artifacts=7):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Artifact prediction branch
        self.artifact_pool = nn.AdaptiveAvgPool2d(1)
        self.artifact_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, num_artifacts),
        )
        self.artifact_projection = nn.Sequential(
            nn.Linear(num_artifacts, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
        )

        # Decoder. Layer names must match the trained checkpoint.
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)

        self.output = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        e4 = self.enc4(p3)
        p4 = self.pool(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Predict multi-label artifact logits and condition bottleneck
        artifact_features = self.artifact_pool(b)
        artifact_logits = self.artifact_head(artifact_features)
        artifact_probs = torch.sigmoid(artifact_logits)
        artifact_embedding = self.artifact_projection(artifact_probs)
        artifact_embedding = artifact_embedding.unsqueeze(-1).unsqueeze(-1)
        b_conditioned = b + artifact_embedding

        # Decoder with skip connections
        d4 = self.up4(b_conditioned)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        restored = torch.sigmoid(self.output(d1))
        return restored, artifact_logits
