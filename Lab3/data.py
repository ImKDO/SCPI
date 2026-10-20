from pathlib import Path

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

LAB_DIR = Path(__file__).parent
DATA_DIR = LAB_DIR / 'data'
TRAIN_DIR = DATA_DIR / 'intel' / 'seg_train' / 'seg_train'
TEST_DIR = DATA_DIR / 'intel' / 'seg_test' / 'seg_test'
CUSTOM_DIR = DATA_DIR / 'custom_test'
WEB_DIR = DATA_DIR / 'web_test'
CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
IMG_SIZE = 150
VAL_FRACTION = 0.15
SEED = 42
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

EVAL_TF = transforms.Compose([
  transforms.Resize((IMG_SIZE, IMG_SIZE)),
  transforms.ToTensor(),
  transforms.Normalize(MEAN, STD),
])
TRAIN_TF = transforms.Compose([
  transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
  transforms.RandomHorizontalFlip(),
  transforms.ColorJitter(0.2, 0.2, 0.2),
  transforms.ToTensor(),
  transforms.Normalize(MEAN, STD),
])


def split_indices():
  targets = datasets.ImageFolder(TRAIN_DIR).targets
  idx = list(range(len(targets)))
  return train_test_split(idx, test_size=VAL_FRACTION, stratify=targets, random_state=SEED)


def make_loaders(augment=True, batch_size=64, workers=0):
  train_idx, val_idx = split_indices()
  train_ds = Subset(datasets.ImageFolder(TRAIN_DIR, TRAIN_TF if augment else EVAL_TF), train_idx)
  val_ds = Subset(datasets.ImageFolder(TRAIN_DIR, EVAL_TF), val_idx)
  kw = {'batch_size': batch_size, 'num_workers': workers, 'persistent_workers': workers > 0}
  return DataLoader(train_ds, shuffle=True, **kw), DataLoader(val_ds, **kw)


def eval_sets():
  train_idx, val_idx = split_indices()
  return {
    'train': Subset(datasets.ImageFolder(TRAIN_DIR, EVAL_TF), train_idx),
    'val': Subset(datasets.ImageFolder(TRAIN_DIR, EVAL_TF), val_idx),
    'seg_test': datasets.ImageFolder(TEST_DIR, EVAL_TF),
    'custom': datasets.ImageFolder(CUSTOM_DIR, EVAL_TF),
    'web': datasets.ImageFolder(WEB_DIR, EVAL_TF),
  }


def device():
  if torch.backends.mps.is_available():
    return torch.device('mps')
  return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
