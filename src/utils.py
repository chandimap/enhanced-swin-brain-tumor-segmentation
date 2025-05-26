import os, time
import numpy as np
import torch

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self):
        self.val=self.sum=self.count=self.avg=0
    def update(self, v, n=1):
        self.val=v; self.sum+=v*n; self.count+=n
        self.avg=self.sum/self.count if self.count else 0

def save_checkpoint(state, cfg, filename="model.pt"):
    out = cfg["output"]["checkpoint_dir"]
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, filename)
    torch.save(state, path)
    print(f"Saved checkpoint: {path}")
