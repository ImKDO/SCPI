import json
import time

import torch
from data import CLASSES, LAB_DIR, device, make_loaders
from models import resnet18
from torch import nn

CKPT_PATH = LAB_DIR / 'checkpoints' / 'resnet18.pt'
RESULTS_DIR = LAB_DIR / 'results'
HISTORY_PATH = RESULTS_DIR / 'history_resnet18.json'
EPOCHS = 10
LR = 3e-4


def run_epoch(model, loader, loss_fn, dev, optimizer=None):
  model.train(optimizer is not None)
  total_loss, correct, n = 0.0, 0, 0
  with torch.set_grad_enabled(optimizer is not None):
    for x, y in loader:
      x, y = x.to(dev), y.to(dev)
      logits = model(x)
      loss = loss_fn(logits, y)
      if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      total_loss += loss.item() * len(y)
      correct += (logits.argmax(1) == y).sum().item()
      n += len(y)
  return total_loss / n, correct / n


def train():
  torch.manual_seed(0)
  dev = device()
  train_loader, val_loader = make_loaders()
  model = resnet18(len(CLASSES)).to(dev)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)
  history, best = [], float('inf')
  CKPT_PATH.parent.mkdir(exist_ok=True)
  for epoch in range(1, EPOCHS + 1):
    t0 = time.time()
    tr_loss, tr_acc = run_epoch(model, train_loader, loss_fn, dev, optimizer)
    va_loss, va_acc = run_epoch(model, val_loader, loss_fn, dev)
    scheduler.step()
    history.append({'epoch': epoch, 'train_loss': tr_loss, 'train_acc': tr_acc, 'val_loss': va_loss, 'val_acc': va_acc})
    mark = ''
    if va_loss < best:
      best = va_loss
      torch.save(model.state_dict(), CKPT_PATH)
      mark = ' *'
    print(f'{epoch:2d}/{EPOCHS} train {tr_loss:.3f}/{tr_acc:.3f}  '
          f'val {va_loss:.3f}/{va_acc:.3f}  {time.time() - t0:.0f}s{mark}', flush=True)
  RESULTS_DIR.mkdir(exist_ok=True)
  HISTORY_PATH.write_text(json.dumps(history, indent=1))


if __name__ == '__main__':
  train()
