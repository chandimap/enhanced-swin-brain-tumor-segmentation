import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR


class HadamardAttention3D(nn.Module):
    """
    Enhanced local self-attention using Hadamard product + a small conv for f(·).
    ŷᵢ = Softmax(conv(Hₖ ⊙ Hq)) · Hv + x
    """
    def __init__(self, channels, kernel_size: int = 3):
        super().__init__()
        # project to Q, K, V
        self.to_q = nn.Conv3d(channels, channels, kernel_size=1)
        self.to_k = nn.Conv3d(channels, channels, kernel_size=1)
        self.to_v = nn.Conv3d(channels, channels, kernel_size=1)
        # f(·) : small local conv
        self.local_conv = nn.Conv3d(channels, channels, kernel_size=kernel_size,
                                    padding=kernel_size//2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        Hq = self.to_q(x)
        Hk = self.to_k(x)
        Hv = self.to_v(x)
        # Hadamard product, local conv, softmax
        attn_logits = self.local_conv(Hk * Hq)
        attn = self.softmax(attn_logits)
        # apply to V and residual
        return attn * Hv + x


class EnhancedLocalSelfAttentionSwinUNETR(nn.Module):
    """
    Hadamard‐attention block on the input features
    to boost local detail representation.
    """
    def __init__(
        self,
        img_size,
        in_channels: int,
        out_channels: int,
        feature_size: int = 48,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        # 1) enhanced local attention on raw input
        self.local_attn = HadamardAttention3D(in_channels)
        # 2) the original SwinUNETR backbone
        self.backbone = SwinUNETR(
            img_size=tuple(img_size),
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            use_checkpoint=use_checkpoint,
        )

    def forward(self, x):
        # Applying Hadamard-based local attention
        x = self.local_attn(x)
        # Feeding into the SwinUNETR
        return self.backbone(x)


def get_model(roi, device):
    """
    Replacing with the enhanced‐attention version.
    """
    model = EnhancedLocalSelfAttentionSwinUNETR(
        img_size=roi,
        in_channels=4,
        out_channels=3,
        feature_size=48,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=True,
    ).to(device)
    return model

