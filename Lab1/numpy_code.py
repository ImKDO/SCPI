import numpy as np


def get_gauss_kernel(size):
  sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8  # та же формула, что в OpenCV
  ax = np.arange(size) - size // 2
  gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
  kernel = np.outer(gauss, gauss)

  return kernel / np.sum(kernel)


def adaptive_threshold_native(image, block_size=11, C=2, method='mean'):
  h, w = image.shape
  offset = block_size // 2

  padded_img = np.pad(image, offset, mode='edge').astype(np.float32)
  local_thresholds = np.zeros((h, w), dtype=np.float32)
  kernel = get_gauss_kernel(block_size)

  for y in range(h):
    for x in range(w):
      window = padded_img[y : y + block_size, x : x + block_size]

      if method == 'mean':
        local_thresholds[y, x] = np.mean(window)
      elif method == 'gauss':
        local_thresholds[y, x] = np.sum(window * kernel)

  final_thresholds = np.round(local_thresholds) - C
  return np.where(image > final_thresholds, 255, 0).astype(np.uint8)
