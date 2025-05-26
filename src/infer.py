import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from functools import partial
from monai.inferers import sliding_window_inference
from src.model import get_model
from src.utils import AverageMeter

def load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model

def infer_case(cfg, case_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg["data"]["roi"], device)
    ckpt = os.path.join(cfg["output"]["checkpoint_dir"], "model.pt")
    load_checkpoint(model, ckpt, device)
    inferer = partial(sliding_window_inference,
                      roi_size=tuple(cfg["data"]["roi"]),
                      sw_batch_size=1,
                      predictor=model, overlap=0.6)
    