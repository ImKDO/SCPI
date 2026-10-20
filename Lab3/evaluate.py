import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from data import CLASSES, device, eval_sets
from matplotlib.colors import LinearSegmentedColormap
from models import resnet18
from PIL import Image
from sklearn.metrics import (
  accuracy_score,
  classification_report,
  confusion_matrix,
  f1_score,
)
from torch.utils.data import DataLoader
from train import CKPT_PATH, HISTORY_PATH, RESULTS_DIR

TRAIN_COLOR, VAL_COLOR = '#2a78d6', '#eb6834'
GOOD, BAD = '#008300', '#d03b3b'
BLUES = LinearSegmentedColormap.from_list('blues', ['#ffffff', '#cde2fb', '#6da7ec', '#2a78d6', '#104281'])
SET_TITLES = {'train': 'train', 'val': 'val (15% seg_train)', 'seg_test': 'seg_test (3000)', 'custom': 'собственный тест (93)', 'web': 'фото из интернета (60)'}

plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True,
                     'grid.color': '#e6e6e6', 'axes.edgecolor': '#999999', 'font.size': 10})


@torch.no_grad()
def predict(model, ds, dev):
  model.eval()
  ys, ps, probs = [], [], []
  for x, y in DataLoader(ds, batch_size=128):
    p = model(x.to(dev)).softmax(1).cpu()
    ys.append(y)
    ps.append(p.argmax(1))
    probs.append(p.max(1).values)
  return torch.cat(ys).numpy(), torch.cat(ps).numpy(), torch.cat(probs).numpy()


def plot_curves(history):
  fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
  ep = [r['epoch'] for r in history]
  best = min(history, key=lambda r: r['val_loss'])['epoch']
  for ax, key, label in zip(axes, ['loss', 'acc'], ['CrossEntropy loss', 'accuracy']):
    ax.plot(ep, [r[f'train_{key}'] for r in history], color=TRAIN_COLOR, lw=2, marker='o', ms=5, label='train')
    ax.plot(ep, [r[f'val_{key}'] for r in history], color=VAL_COLOR, lw=2, marker='o', ms=5, label='val')
    ax.axvline(best, color='#999999', ls='--', lw=1)
    ax.set_xticks(ep)
    ax.set_xlabel('эпоха')
    ax.set_title(label)
    ax.legend(frameon=False)
  fig.suptitle(f'ResNet-18: кривые обучения (пунктир — эпоха {best} с минимальным val loss, она сохранена)')
  fig.tight_layout()
  fig.savefig(RESULTS_DIR / 'curves.png', dpi=130)
  plt.close(fig)


def plot_confusions(preds):
  fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
  for ax, split in zip(axes, ['seg_test', 'custom', 'web']):
    y, p, _ = preds[split]
    cm = confusion_matrix(y, p, labels=range(len(CLASSES)))
    norm = cm / cm.sum(1, keepdims=True)
    ax.imshow(norm, cmap=BLUES, vmin=0, vmax=1)
    for i in range(len(CLASSES)):
      for j in range(len(CLASSES)):
        if cm[i, j]:
          ax.text(j, i, cm[i, j], ha='center', va='center', color='white' if norm[i, j] > 0.6 else '#222222')
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=35, ha='right')
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel('предсказание')
    ax.set_ylabel('истина')
    ax.grid(False)
    ax.set_title(SET_TITLES[split])
  fig.tight_layout()
  fig.savefig(RESULTS_DIR / 'confusion.png', dpi=130)
  plt.close(fig)


def plot_samples(ds, y, p, prob, path, title, only_errors=False, n=24, cols=6):
  idx = np.flatnonzero(y != p) if only_errors else np.random.default_rng(0).permutation(len(y))[:n]
  cols = max(1, min(cols, len(idx)))
  rows = max(1, int(np.ceil(len(idx) / cols)))
  fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.3, rows * 2.7 + 0.4), squeeze=False)
  for ax in axes.flat:
    ax.axis('off')
  for ax, i in zip(axes.flat, idx):
    ax.imshow(Image.open(ds.samples[i][0]).convert('RGB'))
    ok = y[i] == p[i]
    ax.set_title(f'{CLASSES[p[i]]} {prob[i]:.2f}\nистина: {CLASSES[y[i]]}', fontsize=8.5, color=GOOD if ok else BAD)
  fig.suptitle(title)
  fig.tight_layout()
  fig.savefig(path, dpi=110)
  plt.close(fig)


def main():
  dev = device()
  sets = eval_sets()
  model = resnet18(len(CLASSES)).to(dev)
  model.load_state_dict(torch.load(CKPT_PATH, map_location=dev))
  history = json.loads(HISTORY_PATH.read_text())
  preds, metrics = {}, {'best_epoch': min(history, key=lambda r: r['val_loss'])['epoch']}
  for split, ds in sets.items():
    y, p, prob = predict(model, ds, dev)
    preds[split] = (y, p, prob)
    metrics[split] = {'n': len(y), 'acc': accuracy_score(y, p), 'macro_f1': f1_score(y, p, average='macro')}
  (RESULTS_DIR / 'metrics.json').write_text(json.dumps(metrics, indent=1))

  print(f"лучшая эпоха: {metrics['best_epoch']}\n")
  print('| Выборка | Изображений | Accuracy | Macro-F1 |')
  print('|---|---|---|---|')
  for split in sets:
    m = metrics[split]
    print(f"| {SET_TITLES[split]} | {m['n']} | {m['acc']:.3f} | {m['macro_f1']:.3f} |")
  for split in ('seg_test', 'custom', 'web'):
    y, p, _ = preds[split]
    print(f'\n{split}\n' + classification_report(y, p, target_names=CLASSES, digits=3))

  plot_curves(history)
  plot_confusions(preds)
  for split, name in [('custom', 'собственном тесте'), ('web', 'фото из интернета')]:
    y, p, prob = preds[split]
    plot_samples(sets[split], y, p, prob, RESULTS_DIR / f'{split}_samples.png',
                 f'ResNet-18 на {name}: предсказание, уверенность, истина')
    plot_samples(sets[split], y, p, prob, RESULTS_DIR / f'{split}_errors.png',
                 f'Все ошибки на {name}', only_errors=True)


if __name__ == '__main__':
  main()
