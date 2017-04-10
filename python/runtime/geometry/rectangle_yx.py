
import numpy as np
import point_yx as pt

# top, left bottom, right

def __call__(*args, **kwargs): return create(*args, **kwargs)

def create(top_left, bottom_right, dtype=None): return np.array([top_left[0], top_left[1], bottom_right[0], bottom_right[1]], dtype)

def create_with_size(point, size, dtype=None): return create((point[0], point[1]), (point[0]+size[0]-1, point[1]+size[1]-1), dtype)

def centered_rect(center, size, dtype=None):
    return create(
        (center[0] - size[0] / 2, center[1] - size[1] / 2),
        (center[0] + size[0] / 2, center[1] + size[1] / 2))

def top(r): return r[0]

def left(r): return r[1]

def bottom(r): return r[2]

def right(r): return r[3]

def height(r): return 0 if empty(r) else bottom(r) - top(r) + 1

def width(r): return 0 if empty(r) else right(r) - left(r) + 1

def area(r): return width(r) * height(r)

def center(r): return pt.create(top(r)+bottom(r)+1, left(r)+right(r)+1, dtype=r.dtype) / 2

def top_left(r): return pt.create(top(r), left(r))

def top_right(r): return pt.create(top(r), right(r))

def bottom_left(r): return pt.create(bottom(r), left(r))

def bottom_right(r): return pt.create(bottom(r), right(r))

def size(r): return pt.create(height(r), width(r))

def empty(r): return left(r) > right(r) or top(r) > bottom(r)

def intersect(r1, r2):
    return create(
        (np.max(r1[0], r2[0]), np.max(r1[1], r2[1])),
        (np.min(r1[2], r2[2]), np.min(r1[3], r2[3])),
    )

def contains(r, p):
    if pt.x(p) < left(r) or pt.x(p) > right(r) or \
        pt.y(p) < top(r) or pt.y(p) > bottom(r):
        return False
    return True


def clip(r, min, max): return np.clip(r, min, max)


def clip_top_bottom(r, min, max):
    return create(
        (np.clip(r[0], min, max), r[1]),
        (np.clip(r[2], min, max), r[3]),
        r.dtype
    )

def clip_left_right(r, min, max):
    return create(
        (r[0], np.clip(r[1], min, max)),
        (r[2], np.clip(r[3], min, max)),
        r.dtype
    )


def convert_ratio_to_value(rects, size, dtype=None):
    shape = np.shape(rects)
    assert len(shape) <= 2
    w = np.array([size[0]-1, size[1]-1] * 2)
    real = w * rects
    return real.astype(np.int64) if dtype is None else real.astype(dtype)


def convert_value_to_ratio(rects, size, dtype=None):
    w = np.array([size[0] - 1, size[1] - 1] * 2).astype(np.float)
    real = rects / w
    return real.astype(np.float) if dtype is None else real.astype(dtype)



