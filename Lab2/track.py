import argparse
import time
from pathlib import Path

import cv2
from tracker import KeypointTracker, draw


def track_video(src, dst=None, label='object', info=True, show=False, **tracker_args):
  cap = cv2.VideoCapture(str(src))
  ok, frame = cap.read()
  if not ok:
    raise SystemExit(f'не удалось прочитать {src}')
  tracker = KeypointTracker(frame, **tracker_args)
  writer = None
  if dst:
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h))

  log, i = [], 0
  while ok:
    start = time.perf_counter()
    state = tracker.update(frame)
    elapsed = time.perf_counter() - start
    log.append({'frame': i, 'mode': state.mode, 'inliers': state.inliers, 'quad': state.quad, 'time': elapsed})

    text = f'frame {i}  {state.mode}  points {state.inliers}' if info else None
    draw(frame, state, label, text)
    if writer:
      writer.write(frame)
    if show:
      cv2.imshow('tracking', frame)
      if cv2.waitKey(1) & 0xFF in (27, ord('q')):
        break
    ok, frame = cap.read()
    i += 1

  if writer:
    writer.release()
  if show:
    cv2.destroyAllWindows()
  return log


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Трекинг объекта с первого кадра по ключевым точкам')
  parser.add_argument('video')
  parser.add_argument('-o', '--output', help='куда сохранить видео с рамкой (.mp4)')
  parser.add_argument('--label', default='object')
  parser.add_argument('--detector', choices=['sift', 'orb'], default='sift')
  parser.add_argument('--no-flow', action='store_true', help='только детекция, без оптического потока')
  parser.add_argument('--roi', help='область объекта на первом кадре: x,y,w,h (по умолчанию весь кадр)')
  parser.add_argument('--show', action='store_true', help='показывать кадры в окне')
  args = parser.parse_args()
  roi = tuple(int(v) for v in args.roi.split(',')) if args.roi else None

  output = args.output or Path(args.video).with_name(Path(args.video).stem + '_tracked.mp4')
  log = track_video(args.video, output, args.label, show=args.show,
                    roi=roi, detector=args.detector, use_flow=not args.no_flow)
  modes = [r['mode'] for r in log]
  total = sum(r['time'] for r in log)
  print(f'кадров: {len(log)}, detect: {modes.count("detect")}, flow: {modes.count("flow")}, '
        f'lost: {modes.count("lost")}, {len(log) / total:.0f} кадр/с')
  print(f'результат: {output}')
