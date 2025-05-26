import torch
from monai.networks.nets import SwinUNETR

def get_model(roi, device):
    model = SwinUNETR(
        img_size=tuple(roi),
        in_channels=4, out_channels=3,
        feature_size=48, drop_rate=0.,
        attn_drop_rate=0., dropout_path_rate=0.,
        use_checkpoint=True
    ).to(device)
    return model
