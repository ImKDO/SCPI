from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from experiments import iou
from track import track_video

LAB_DIR = Path(__file__).parent
VIDEO = LAB_DIR / 'videos' / 'mona-lisa-blur-extra-credit.avi'
REFERENCE = LAB_DIR / 'videos' / 'mona-lisa.avi'
DEFAULTS = {'ratio': 0.75, 'min_inliers': 15, 'ransac_thresh': 3.0, 'fb_thresh': 2.0, 'max_hold': 5, 'max_points': 200}
SWEEPS = {
  'ratio': [0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95],
  'min_inliers': [4, 8, 15, 30, 60],
  'ransac_thresh': [0.5, 1, 3, 10, 30],
  'fb_thresh': [0.2, 0.5, 2, 10, 1000],
  'max_hold': [0, 1, 5, 30],
  'max_points': [10, 30, 200, 1000],
}


def main():
  ref = [r['quad'] for r in track_video(REFERENCE, info=False)]
  shape = cv2.VideoCapture(str(REFERENCE)).read()[1].shape[:2]
  blurred = list(range(10, 29)) + list(range(391, 400))
  results = {}

  print('| Параметр | Значение | detect | flow | hold | lost | IoU (все) | IoU (размытые) | кадр/с |')
  print('|---|---|---|---|---|---|---|---|---|')
  for name, values in SWEEPS.items():
    for value in values:
      log = track_video(VIDEO, info=False, **{**DEFAULTS, name: value})
      modes = [r['mode'] for r in log]
      scores = [iou(r['quad'], q, shape) for r, q in zip(log, ref)]
      fps = len(log) / sum(r['time'] for r in log)
      results.setdefault(name, []).append((value, np.mean(scores), np.mean([scores[i] for i in blurred])))
      mark = ' (по умолч.)' if value == DEFAULTS[name] else ''
      print(f'| {name} | {value}{mark} | {modes.count("detect")} | {modes.count("flow")} | {modes.count("hold")} '
            f'| {modes.count("lost")} | {np.mean(scores):.3f} | {results[name][-1][2]:.3f} | {fps:.0f} |')

  fig, axes = plt.subplots(2, 3, figsize=(12, 6))
  for ax, (name, rows) in zip(axes.ravel(), results.items()):
    x = np.arange(len(rows))
    ax.plot(x, [r[1] for r in rows], 'o-', label='все кадры')
    ax.plot(x, [r[2] for r in rows], 's--', label='размытые кадры')
    ax.set_xticks(x, [str(r[0]) for r in rows])
    ax.set_title(name, fontsize=10)
    ax.set_ylim(-0.05, 1.05)
  axes[0, 0].set_ylabel('IoU с эталоном')
  axes[1, 0].set_ylabel('IoU с эталоном')
  axes[0, 0].legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(LAB_DIR / 'results' / 'params.png', dpi=120)


if __name__ == '__main__':
  main()
