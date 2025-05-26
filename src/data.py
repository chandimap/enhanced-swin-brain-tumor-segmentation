import os, json
from monai import data, transforms
from monai.transforms import (
    LoadImaged, ConvertToMultiChannelBasedOnBratsClassesd,
    CropForegroundd, RandSpatialCropd, RandFlipd,
    NormalizeIntensityd, RandScaleIntensityd, RandShiftIntensityd
)

def read_splits(json_path, basedir, fold, key="training"):
    with open(json_path) as f:
        items = json.load(f)[key]
    for item in items:
        for k,v in item.items():
            if isinstance(v, list):
                item[k] = [os.path.join(basedir, p) for p in v]
            elif isinstance(v, str) and v:
                item[k] = os.path.join(basedir, v)
    tr, val = [], []
    for d in items:
        if d.get("fold", -1) == fold:
            val.append(d)
        else:
            tr.append(d)
    return tr, val

def get_loaders(cfg):
    tr_files, val_files = read_splits(
        cfg["data"]["json_splits"], cfg["data"]["root_dir"], cfg["data"]["fold"]
    )
    roi = tuple(cfg["data"]["roi"])
    train_trans = transforms.Compose([
        LoadImaged(keys=["image","label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        CropForegroundd(keys=["image","label"], source_key="image", k_divisible=roi),
        RandSpatialCropd(keys=["image","label"], roi_size=roi, random_size=False),
        RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=2),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
    ])
    val_trans = transforms.Compose([
        LoadImaged(keys=["image","label"]),
        ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    ])
    train_ds = data.Dataset(data=tr_files, transform=train_trans)
    val_ds = data.Dataset(data=val_files, transform=val_trans)
    train_loader = data.DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                                   shuffle=True, num_workers=8, pin_memory=True)
    val_loader   = data.DataLoader(val_ds, batch_size=1,
                                   shuffle=False, num_workers=8, pin_memory=True)
    return train_loader, val_loader
