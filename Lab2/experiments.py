from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from track import track_video

LAB_DIR = Path(__file__).parent
VIDEOS_DIR = LAB_DIR / 'videos'
RESULTS_DIR = LAB_DIR / 'results'
TEST_VIDEOS = ['mona-lisa', 'mona-lisa-blur', 'mona-lisa-blur-extra-credit']
SAMPLE = 'simple_track'
LABELS = {'mona-lisa': 'Mona Lisa', 'self': 'Wallet'}
ROIS = {'self': (90, 278, 541, 749)}
CONFIGS = {
  'SIFT': {'detector': 'sift', 'use_flow': False},
  'SIFT + LK': {'detector': 'sift', 'use_flow': True},
  'ORB': {'detector': 'orb', 'use_flow': False},
  'ORB + LK': {'detector': 'orb', 'use_flow': True},
}
MODE_COLORS = {'detect': '#2ca02c', 'flow': '#f0a500', 'hold': '#d62728', 'lost': '#cccccc'}


def label_for(name):
  return next((v for k, v in LABELS.items() if name.startswith(k)), name)


def iou(a, b, shape):
  if a is None or b is None:
    return 0.0
  ma, mb = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
  cv2.fillConvexPoly(ma, a.astype(np.int32), 1)
  cv2.fillConvexPoly(mb, b.astype(np.int32), 1)
  return (ma & mb).sum() / max((ma | mb).sum(), 1)


def read_frames(path, indices, size=None):
  cap = cv2.VideoCapture(str(path))
  frames = {}
  for i in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, frame = cap.read()
    if ok:
      frames[i] = cv2.resize(frame, size) if size else frame
  return frames


def frames_grid(path, indices, cols=4, size=(320, 180)):
  frames = read_frames(path, indices, size)
  tiles = [frames[i] for i in indices if i in frames]
  tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % cols)
  return np.vstack([np.hstack(tiles[r:r + cols]) for r in range(0, len(tiles), cols)])


def save_gif(path, dst, step=3, size=(288, 162)):
  cap = cv2.VideoCapture(str(path))
  fps = cap.get(cv2.CAP_PROP_FPS) or 30
  images, i = [], 0
  while True:
    ok, frame = cap.read()
    if not ok:
      break
    if i % step == 0:
      images.append(Image.fromarray(cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2RGB)).quantize(48))
    i += 1
  images[0].save(dst, save_all=True, append_images=images[1:], duration=int(1000 * step / fps), loop=0, optimize=True)


def main():
  RESULTS_DIR.mkdir(exist_ok=True)
  own = sorted(p.stem for p in VIDEOS_DIR.iterdir()
               if p.suffix.lower() in ('.avi', '.mp4', '.mov') and p.stem not in TEST_VIDEOS + [SAMPLE] and not p.stem.endswith('_tracked'))
  videos = TEST_VIDEOS + own

  # основной прогон: SIFT + LK, видео с рамкой
  logs = {}
  for name in videos:
    src = next(VIDEOS_DIR.glob(name + '.*'))
    logs[name] = track_video(src, RESULTS_DIR / f'{name}_tracked.mp4', label_for(name), roi=ROIS.get(name))
  shape = cv2.VideoCapture(str(VIDEOS_DIR / 'mona-lisa.avi')).read()[1].shape[:2]
  ref = [r['quad'] for r in logs['mona-lisa']]

  print('| Видео | Кадров | detect | flow | hold | lost | Рамка есть | IoU с эталоном | кадр/с |')
  print('|---|---|---|---|---|---|---|---|---|')
  for name in videos:
    log = logs[name]
    modes = [r['mode'] for r in log]
    fps = len(log) / sum(r['time'] for r in log)
    shown = 100 * (1 - modes.count('lost') / len(log))
    score = '—' if name not in TEST_VIDEOS[1:] else \
      f'{np.mean([iou(r["quad"], q, shape) for r, q in zip(log, ref)]):.3f}'
    print(f'| {name} | {len(log)} | {modes.count("detect")} | {modes.count("flow")} | {modes.count("hold")} '
          f'| {modes.count("lost")} | {shown:.1f}% | {score} | {fps:.0f} |')

  # сравнение конфигураций
  print('\n| Конфигурация | ' + ' | '.join(TEST_VIDEOS) + ' | кадр/с |')
  print('|---|' + '---|' * (len(TEST_VIDEOS) + 1))
  config_logs = {}
  for cname, args in CONFIGS.items():
    cells, times = [], []
    for name in TEST_VIDEOS:
      log = logs[name] if cname == 'SIFT + LK' else track_video(VIDEOS_DIR / f'{name}.avi', info=False, **args)
      config_logs[cname, name] = log
      shown = 100 * np.mean([r['mode'] != 'lost' for r in log])
      score = np.mean([iou(r['quad'], q, shape) for r, q in zip(log, ref)])
      cells.append(f'{shown:.0f}% / {score:.2f}')
      times.append(len(log) / sum(r['time'] for r in log))
    print(f'| {cname} | ' + ' | '.join(cells) + f' | {np.mean(times):.0f} |')

  blurred = list(range(10, 29)) + list(range(391, 400))
  print('\nIoU на размытых кадрах:')
  for cname in CONFIGS:
    for name in TEST_VIDEOS[1:]:
      frames = blurred if 'extra' in name else blurred[:19]
      log = config_logs[cname, name]
      print(f'  {cname}, {name}: {np.mean([iou(log[i]["quad"], ref[i], shape) for i in frames]):.3f}')

  # дрейф: детекция только на первом кадре, дальше только оптический поток
  drift_log = track_video(VIDEOS_DIR / 'mona-lisa.avi', info=False, min_inliers=10 ** 6)
  reliable = [i for i, r in enumerate(logs['mona-lisa']) if r['mode'] == 'detect' and r['inliers'] >= 30]
  drift = [np.linalg.norm(drift_log[i]['quad'] - ref[i], axis=1).mean() if drift_log[i]['quad'] is not None
           else np.nan for i in reliable]
  print('\nдрейф (только LK), px:', {i: round(d, 1) for i, d in zip(reliable[::20], drift[::20])})

  # временная шкала режимов и числа точек
  fig, axes = plt.subplots(len(videos), 1, figsize=(11, 1.8 * len(videos)), sharex=True, squeeze=False)
  for ax, name in zip(axes[:, 0], videos):
    log = logs[name]
    for r in log:
      ax.axvspan(r['frame'] - 0.5, r['frame'] + 0.5, color=MODE_COLORS[r['mode']], alpha=0.35, lw=0)
    ax.plot([r['inliers'] for r in log], color='#333333', lw=1)
    ax.set_yscale('symlog', linthresh=10)
    ax.set_ylim(0, 500)
    ax.set_ylabel('точек')
    ax.set_title(name, fontsize=10, loc='left')
  handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.5) for c in MODE_COLORS.values()]
  axes[0, 0].legend(handles, MODE_COLORS.keys(), ncol=4, loc='upper right', fontsize=8)
  axes[-1, 0].set_xlabel('кадр')
  fig.tight_layout()
  fig.savefig(RESULTS_DIR / 'timeline.png', dpi=120)

  # IoU на размытых видео
  fig, ax = plt.subplots(figsize=(11, 3.2))
  for cname, color in [('SIFT', '#9467bd'), ('SIFT + LK', '#1f77b4')]:
    for name, style in [('mona-lisa-blur', '-'), ('mona-lisa-blur-extra-credit', ':')]:
      log = config_logs[cname, name]
      ax.plot([iou(r['quad'], q, shape) for r, q in zip(log, ref)], style, color=color, lw=1.2,
              label=f'{cname}, {name}')
  ax.set_xlabel('кадр')
  ax.set_ylabel('IoU с эталоном')
  ax.set_ylim(-0.05, 1.05)
  ax.legend(fontsize=8, loc='lower right')
  fig.tight_layout()
  fig.savefig(RESULTS_DIR / 'iou_blur.png', dpi=120)

  # дрейф
  fig, ax = plt.subplots(figsize=(11, 3))
  ax.plot(reliable, drift, '.', color='#d62728')
  ax.set_xlabel('кадр')
  ax.set_ylabel('ошибка углов, px')
  ax.set_title('Только оптический поток: отклонение от детекции SIFT (≥30 inliers)', fontsize=10)
  fig.tight_layout()
  fig.savefig(RESULTS_DIR / 'drift.png', dpi=120)

  # кадры с результатами
  keyframes = [0, 60, 120, 200, 280, 384, 470, 556]
  for name in videos:
    n = len(logs[name])
    if name in TEST_VIDEOS:
      grid = frames_grid(RESULTS_DIR / f'{name}_tracked.mp4', keyframes)
    else:
      idx = list(np.linspace(0, n - 1, 12).astype(int))
      grid = frames_grid(RESULTS_DIR / f'{name}_tracked.mp4', idx, cols=6, size=(180, 320))
    cv2.imwrite(str(RESULTS_DIR / f'{name}_frames.jpg'), grid)
  for name in ['mona-lisa-blur', 'mona-lisa-blur-extra-credit']:
    idx = [9, 12, 18, 26, 30, 392, 396, 400] if 'extra' in name else [9, 11, 14, 17, 20, 23, 26, 30]
    cv2.imwrite(str(RESULTS_DIR / f'{name}_blur_frames.jpg'), frames_grid(RESULTS_DIR / f'{name}_tracked.mp4', idx))

  # сравнение с образцом
  idx = [139, 200, 278, 417]
  sample = frames_grid(VIDEOS_DIR / f'{SAMPLE}.avi', idx)
  ours = frames_grid(RESULTS_DIR / 'mona-lisa_tracked.mp4', idx)
  cv2.imwrite(str(RESULTS_DIR / 'sample_vs_ours.jpg'), np.vstack([sample, ours]))

  save_gif(RESULTS_DIR / 'mona-lisa_tracked.mp4', RESULTS_DIR / 'mona-lisa.gif')
  for name in own:
    save_gif(RESULTS_DIR / f'{name}_tracked.mp4', RESULTS_DIR / f'{name}.gif', step=6, size=(162, 288))


if __name__ == '__main__':
  main()
