from dataclasses import dataclass

import cv2
import numpy as np

LK_PARAMS = {'winSize': (21, 21), 'maxLevel': 4,
             'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)}
COLORS = {'detect': (0, 200, 0), 'flow': (0, 200, 255), 'hold': (0, 0, 255)}


@dataclass
class State:
  mode: str
  quad: np.ndarray | None = None
  points: np.ndarray | None = None
  inliers: int = 0


def to_gray(frame):
  return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame


class KeypointTracker:
  def __init__(self, template, roi=None, detector='sift', use_flow=True, min_inliers=15, ratio=0.75,
               max_points=200, max_hold=5, ransac_thresh=3.0, fb_thresh=2.0):
    self.use_flow = use_flow
    self.max_hold = max_hold
    self.misses = 0
    self.min_inliers = min_inliers
    self.ratio = ratio
    self.max_points = max_points
    self.ransac_thresh = ransac_thresh
    self.fb_thresh = fb_thresh
    if detector == 'sift':
      self.detector = cv2.SIFT_create()
      self.matcher = cv2.FlannBasedMatcher({'algorithm': 1, 'trees': 5}, {'checks': 50})
    else:
      self.detector = cv2.ORB_create(nfeatures=3000)
      self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    gray = to_gray(template)
    x, y, w, h = roi or (0, 0, gray.shape[1], gray.shape[0])
    self.corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]).reshape(-1, 1, 2)
    mask = np.zeros_like(gray)
    mask[y:y + h, x:x + w] = 255
    kp, self.desc = self.detector.detectAndCompute(gray, mask)
    self.template_pts = np.float32([k.pt for k in kp])
    self.H = np.eye(3)
    self.prev = gray
    self.seed_points(gray)

  def quad(self, H):
    return cv2.perspectiveTransform(self.corners, H).reshape(-1, 2)

  def is_valid(self, H):
    quad = self.quad(H)
    if not cv2.isContourConvex(quad) or cv2.contourArea(quad) < 400:
      return False
    sides = np.linalg.norm(quad - np.roll(quad, 1, axis=0), axis=1)
    return sides.max() / sides.min() < 6

  def detect(self, gray):
    kp, desc = self.detector.detectAndCompute(gray, None)
    if desc is None or len(kp) < self.min_inliers:
      return None
    pairs = self.matcher.knnMatch(self.desc, desc, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < self.ratio * p[1].distance]
    if len(good) < self.min_inliers:
      return None
    src = self.template_pts[[m.queryIdx for m in good]]
    dst = np.float32([kp[m.trainIdx].pt for m in good])
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_thresh)
    if H is None or mask.sum() < self.min_inliers or not self.is_valid(H):
      return None
    return H, dst[mask.ravel() == 1]

  def seed_points(self, gray):
    mask = np.zeros_like(gray)
    cv2.fillConvexPoly(mask, self.quad(self.H).astype(np.int32), 255)
    pts = cv2.goodFeaturesToTrack(gray, self.max_points, 0.001, 5, mask=mask)
    self.points = pts
    self.anchors = None if pts is None else cv2.perspectiveTransform(pts, np.linalg.inv(self.H))

  def track(self, gray):
    if self.H is None or self.points is None or len(self.points) < 8:
      return None
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.points, None, **LK_PARAMS)
    back, st_back, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, nxt, None, **LK_PARAMS)
    fb_error = np.linalg.norm(self.points - back, axis=2).ravel()
    ok = (st.ravel() == 1) & (st_back.ravel() == 1) & (fb_error < self.fb_thresh)
    if ok.sum() < 8:
      return None
    H, mask = cv2.findHomography(self.anchors[ok], nxt[ok], cv2.RANSAC, self.ransac_thresh)
    if H is None or not self.is_valid(H):
      return None
    keep = ok.copy()
    keep[ok] = mask.ravel() == 1
    self.points, self.anchors = nxt[keep], self.anchors[keep]
    return H, self.points.reshape(-1, 2)

  def update(self, frame):
    gray = to_gray(frame)
    mode, result = 'detect', self.detect(gray)
    if result is None and self.use_flow:
      mode, result = 'flow', self.track(gray)

    if result is None and self.use_flow and self.H is not None and self.misses < self.max_hold:
      self.misses += 1
      self.seed_points(gray)
      state = State('hold', self.quad(self.H), np.empty((0, 2)))
    elif result is None:
      self.H = None
      state = State('lost')
    else:
      self.misses = 0
      self.H, points = result
      if mode == 'detect' or len(self.points) < self.max_points // 2:
        self.seed_points(gray)
      state = State(mode, self.quad(self.H), points, len(points))
    self.prev = gray
    return state


def draw(frame, state, label, info=None):
  if state.quad is not None:
    color = COLORS[state.mode]
    for x, y in state.points:
      cv2.circle(frame, (int(x), int(y)), 1, color, -1)
    cv2.polylines(frame, [state.quad.astype(np.int32)], True, color, 2, cv2.LINE_AA)

    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    h, w = frame.shape[:2]
    x, y = state.quad[np.argmin(state.quad[:, 1])]
    x = int(np.clip(x - tw / 2, 0, w - tw - 6))
    y = int(np.clip(y - 8, th + base + 4, h - 2))
    cv2.rectangle(frame, (x, y - th - base - 4), (x + tw + 6, y), color, -1)
    cv2.putText(frame, label, (x + 3, y - base - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

  if info:
    cv2.putText(frame, info, (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
  return frame
