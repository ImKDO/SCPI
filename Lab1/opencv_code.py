import cv2

METHODS = {
  'mean': cv2.ADAPTIVE_THRESH_MEAN_C,
  'gauss': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
}


def adaptive_threshold_cv(image, block_size=11, C=2, method='mean'):
  return cv2.adaptiveThreshold(image, 255, METHODS[method], cv2.THRESH_BINARY, block_size, C)
