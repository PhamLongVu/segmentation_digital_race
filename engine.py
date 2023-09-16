import torch
from config import config

class engine():
  def train_batch(model, data, optimizer, criterion):
    model.train()
    ims, ce_masks = data
    ims = ims.to(config.DEVICE).float()
    ce_masks = ce_masks.to(config.DEVICE).float()
    ce_masks = ce_masks.permute(0, 3, 1, 2).contiguous().view(-1, 3, 480, 640)
    _masks = model(ims)
    optimizer.zero_grad()
    loss= criterion(_masks.squeeze(1), ce_masks)
    loss.backward()
    optimizer.step()
    return loss.item()

  @torch.no_grad()
  def validate_batch(model, data, criterion):
    model.eval()
    with torch.no_grad():
        ims, masks = data
        ims = ims.to(config.DEVICE).float()
        masks = masks.to(config.DEVICE).float()
        masks = masks.permute(0, 3, 1, 2).contiguous().view(-1, 3, 480, 640)
        _masks = model(ims)

        loss= criterion(_masks.squeeze(1), masks)

        return loss.item()