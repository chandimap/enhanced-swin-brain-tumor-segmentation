import os, time
import torch, numpy as np
from functools import partial
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.inferers import sliding_window_inference
from monai.transforms import Activations, AsDiscrete
from src.data import get_loaders
from src.model import get_model
from src.utils import AverageMeter, save_checkpoint

def train_epoch(model, loader, optimizer, loss_fn, meter, device):
    model.train(); meter.reset()
    start = time.time()
    for i, batch in enumerate(loader):
        imgs = batch["image"].to(device); lbls = batch["label"].to(device)
        optimizer.zero_grad()
        out = model(imgs); loss = loss_fn(out, lbls)
        loss.backward(); optimizer.step()
        meter.update(loss.item(), n=imgs.shape[0])
        print(f"[Train] {i+1}/{len(loader)} loss={meter.avg:.4f} time={(time.time()-start):.2f}s")
        start=time.time()
    return meter.avg

def val_epoch(model, loader, inferer, metric, post_ops, device):
    model.eval(); metric.reset()
    meter = AverageMeter()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            imgs = batch["image"].to(device); lbls = batch["label"].to(device)
            logits = inferer(imgs)
            preds = post_ops(logits)
            metric(y_pred=preds, y= [lbls])
            acc, _ = metric.aggregate()
            meter.update(acc.cpu().item())
            print(f"[Val] {i+1}/{len(loader)} dice={meter.avg:.4f}")
    return meter.avg

def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = get_loaders(cfg)
    model = get_model(cfg["data"]["roi"], device)
    loss_fn = DiceLoss(sigmoid=True)
    post = lambda x: AsDiscrete(argmax=False, threshold=0.5)(Activations(sigmoid=True)(x))
    metric = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH)
    inferer = partial(sliding_window_inference,
                      roi_size=tuple(cfg["data"]["roi"]),
                      sw_batch_size=cfg["training"]["sw_batch_size"],
                      predictor=model, overlap=cfg["inference"]["overlap"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"],
                                  weight_decay=cfg["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["max_epochs"])
    best = 0
    for epoch in range(cfg["training"]["max_epochs"]):
        print(f"Epoch [{epoch}/{cfg['training']['max_epochs']}]")
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, AverageMeter(), device)
        if epoch % cfg["training"]["val_every"] == 0:
            val_dice = val_epoch(model, val_loader, inferer, metric, post, device)
            if val_dice > best:
                best = val_dice
                save_checkpoint({"epoch": epoch, "state_dict": model.state_dict(), "best": best}, cfg)
        scheduler.step()
    print(f"Training done. Best Dice: {best:.4f}")

if __name__=="__main__":
    import yaml, sys
    cfg = yaml.safe_load(open(sys.argv[1]))
    main(cfg)
