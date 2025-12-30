import torch
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.nn.functional as F

class Conv3x3BNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

class DecoderBlock(nn.Module):
    """_summary_
    Upsample (bilinear) -> concat skip -> two 3x3 convs
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = Conv3x3BNReLU(in_ch + skip_ch, out_ch)
        self.conv2 = Conv3x3BNReLU(out_ch, out_ch)

    def forward(self, x, skip):
        x = self.upsample(x)
        # pad on odd shapes
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class ResNet50UNet(nn.Module):
    def __init__(self, encoder, num_classes: int = 1):
        """_summary_
        encoder: a ResNet50-like model instance (torchvision/torchgeo resnet).
        num_classes: number of output classes (logits).
        encoder_pretrained: if True, encoder params are left as-is; set requires_grad outside if needed.
        """
        super().__init__()
        self.encoder = encoder

        # expect ResNet50 channel sizes
        enc_channels = dict(
            c1 = 64,    # after conv1
            c2 = 256,   # layer1
            c3 = 512,   # layer2
            c4 = 1024,  # layer3
            c5 = 2048,  # layer4 (bottleneck)
        )

        # Decoder: progressively reduce channels
        self.dec4 = DecoderBlock(in_ch=enc_channels["c5"], skip_ch=enc_channels["c4"], out_ch=512)
        self.dec3 = DecoderBlock(in_ch=512, skip_ch=enc_channels["c3"], out_ch=256)
        self.dec2 = DecoderBlock(in_ch=256, skip_ch=enc_channels["c2"], out_ch=128)
        self.dec1 = DecoderBlock(in_ch=128, skip_ch=enc_channels["c1"], out_ch=64)

        # segmentation head
        self.head = nn.Sequential(
            Conv3x3BNReLU(64, 64),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

        # init decoder weights
        self._init_weights()

        self.refine = Conv3x3BNReLU(num_classes, num_classes)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """_summary_
        Iterate through the UNet by encoder-decoder architecture. 
        """
        inp_h, inp_w = x.shape[-2], x.shape[-1]
        # encoder forward (assumes usual ResNet flow)
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = F.relu(x)
        c1 = x          # low-level features (64 channels)
        x = self.encoder.maxpool(x)

        # Using the learning layers in the encoder
        c2 = self.encoder.layer1(x)  # 256
        c3 = self.encoder.layer2(c2) # 512
        c4 = self.encoder.layer3(c3) # 1024
        c5 = self.encoder.layer4(c4) # 2048 (bottleneck)

        # decoder (UNet-style)
        d4 = self.dec4(c5, c4)
        d3 = self.dec3(d4, c3)
        d2 = self.dec2(d3, c2)
        d1 = self.dec1(d2, c1)

        logits = self.head(d1)

        if logits.shape[-2:] != (inp_h, inp_w):
            logits = F.interpolate(logits, size=(inp_h, inp_w), mode="bilinear", align_corners=False)
            logits = self.refine(logits)
            
        return logits

def build_unet_from_resnet50(resnet50_model, num_classes: int = 11, unfreeze: bool = False):
    """_summary_
    Convenience layer that wraps a given resnet50 (e.g. resnet50(weights)) into a UNet head.
    """
    net = ResNet50UNet(resnet50_model, num_classes=num_classes)
    for p in resnet50_model.parameters():
        p.requires_grad = False

    # unfreeze last block for improved performance
    if unfreeze:
        for p in resnet50_model.layer4.parameters():
            p.requires_grad = True

    return net