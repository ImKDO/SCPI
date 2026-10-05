import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy_code import adaptive_threshold_native
from opencv_code import adaptive_threshold_cv

LAB_DIR = Path(__file__).parent
RESULTS_DIR = LAB_DIR / 'results'
BLOCK_SIZE = 11
C = 2


def measure(func, image, method, repeats=1, block_size=BLOCK_SIZE):
  best = float('inf')
  for _ in range(repeats):
    start = time.perf_counter()
    result = func(image, block_size, C, method)
    best = min(best, time.perf_counter() - start)
  return result, best


RESULTS_DIR.mkdir(exist_ok=True)
labels, cv_times, native_times = [], [], []

print('| Изображение | Метод | OpenCV, мс | Нативно, мс | OpenCV быстрее в | Совпадение |')
print('|---|---|---|---|---|---|')

for path in sorted(LAB_DIR.glob('*.jpg')):
  image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
  h, w = image.shape
  outputs = [image]

  for method in ('mean', 'gauss'):
    cv_res, cv_time = measure(adaptive_threshold_cv, image, method, repeats=10)
    native_res, native_time = measure(adaptive_threshold_native, image, method)
    match = np.mean(cv_res == native_res) * 100

    outputs.append(cv_res)
    labels.append(f'{path.stem}\n{method}')
    cv_times.append(cv_time * 1000)
    native_times.append(native_time * 1000)
    print(f'| {path.name} ({w}x{h}) | {method} | {cv_time * 1000:.2f} | {native_time * 1000:.0f} '
          f'| x{native_time / cv_time:.0f} | {match:.3f}% |')

  # исходное | mean | gauss
  cv2.imwrite(str(RESULTS_DIR / f'{path.stem}_result.png'), np.hstack(outputs))

x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - 0.2, cv_times, 0.4, label='OpenCV', color='#2a78d6')
ax.bar(x + 0.2, native_times, 0.4, label='Нативно (NumPy + циклы)', color='#eb6834')
ax.set_xticks(x, labels)
ax.set_yscale('log')
ax.set_ylabel('время, мс (лог. шкала)')
ax.set_title(f'Время адаптивной бинаризации (b={BLOCK_SIZE}, C={C})')
ax.legend()
fig.tight_layout()
fig.savefig(RESULTS_DIR / 'time.png', dpi=120)


# ---- Зависимость от размера изображения и от размера окна ----

IMPLEMENTATIONS = {
  'OpenCV': (adaptive_threshold_cv, 10, '#2a78d6'),
  'Нативно': (adaptive_threshold_native, 1, '#eb6834'),
}
LINE_STYLES = {'mean': '-', 'gauss': '--'}


def plot_lines(x, times, xlabel, title, filename, logx=False):
  fig, ax = plt.subplots(figsize=(9, 5))
  for (name, method), values in times.items():
    color = IMPLEMENTATIONS[name][2]
    ax.plot(x, values, LINE_STYLES[method], color=color, marker='o', label=f'{name}, {method}')
  if logx:
    ax.set_xscale('log')
  ax.set_yscale('log')
  ax.set_xlabel(xlabel)
  ax.set_ylabel('время, мс (лог. шкала)')
  ax.set_title(title)
  ax.grid(True, alpha=0.3)
  ax.legend()
  fig.tight_layout()
  fig.savefig(RESULTS_DIR / filename, dpi=120)


base = cv2.imread(str(LAB_DIR / '2.jpg'), cv2.IMREAD_GRAYSCALE)

sizes = [64, 128, 256, 512, 1024]
size_times = {}
for name, (func, repeats, _) in IMPLEMENTATIONS.items():
  for method in ('mean', 'gauss'):
    size_times[name, method] = []
    for side in sizes:
      image = cv2.resize(base, (side, side))
      size_times[name, method].append(measure(func, image, method, repeats)[1] * 1000)
    print(f'размер {name} {method}:', [f'{t:.2f}' for t in size_times[name, method]])

megapixels = [side * side / 1e6 for side in sizes]
plot_lines(megapixels, size_times, 'мегапиксели (лог. шкала)',
           f'Время от размера изображения (b={BLOCK_SIZE})', 'time_vs_size.png', logx=True)

blocks = [3, 7, 11, 21, 31, 51]
small = cv2.resize(base, (256, 256))
block_times = {}
for name, (func, repeats, _) in IMPLEMENTATIONS.items():
  for method in ('mean', 'gauss'):
    block_times[name, method] = [
      measure(func, small, method, repeats, block_size=b)[1] * 1000 for b in blocks
    ]
    print(f'окно {name} {method}:', [f'{t:.2f}' for t in block_times[name, method]])

plot_lines(blocks, block_times, 'размер окна b', 'Время от размера окна (256x256)', 'time_vs_block.png')
