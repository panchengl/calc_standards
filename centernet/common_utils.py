import numpy as np
def affine_transform(pt, t):
  new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
  new_pt = np.dot(t, new_pt)
  return new_pt[:2]

