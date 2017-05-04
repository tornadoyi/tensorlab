import tensorflow as tf
import point_yx as pt


def create(tl, br, dtype=None):
    assert tl.shape.ndims == br.shape.ndims

    ndims = tl.shape.ndims
    assert ndims == 1 or ndims == 2

    if dtype is None:
        return tf.convert_to_tensor(tf.concat([tl, br], axis=0 if len(tl.shape) == 1 else 1), dtype)
    else:
        return tf.cast(tf.concat([tl, br], axis=0 if len(tl.shape) == 1 else 1), dtype)

def create_with_size(point, size, dtype=None): return create(point, point+size-1, dtype)

def centered_rect(center, size, dtype=None): return create_with_size(center - (size-1)/2, size, dtype)

def top(r): return r[0] if len(r.shape) == 1 else r[:,0]

def left(r): return r[1] if len(r.shape) == 1  else r[:,1]

def bottom(r): return r[2] if len(r.shape) == 1  else r[:,2]

def right(r): return r[3] if len(r.shape) == 1  else r[:,3]

def height(r): b,t = bottom(r), top(r); return tf.maximum(b-t+1, 0)

def width(r): r,l = right(r), left(r); return tf.maximum(r-l+1, 0)

def area(r): return width(r) * height(r)

def center(r): return pt.create(top(r)+bottom(r)+1, left(r)+right(r)+1, dtype=r.dtype) / 2

def top_left(r): return pt.create(top(r), left(r))

def top_right(r): return pt.create(top(r), right(r))

def bottom_left(r): return pt.create(bottom(r), left(r))

def bottom_right(r): return pt.create(bottom(r), right(r))

def size(r): return pt.create(height(r), width(r))

def empty(r): return tf.logical_or(left(r) > right(r), top(r) > bottom(r))

def intersect(r1, r2):
    return create(
        pt.create(tf.maximum(top(r1), top(r2)), tf.maximum(left(r1), left(r2))),
        pt.create(tf.minimum(bottom(r1), bottom(r2)), tf.minimum(right(r1), right(r2))),
    )


def union(r1, r2):
    return create(
        pt.create(tf.minimum(top(r1), top(r2)), tf.minimum(left(r1), left(r2))),
        pt.create(tf.maximum(bottom(r1), bottom(r2)), tf.maximum(right(r1), right(r2)))
    )


def contains(r, p):
    cond = tf.convert_to_tensor(False)
    cond = tf.logical_or(cond, pt.x(p) < left(r))
    cond = tf.logical_or(cond, pt.x(p) > right(r))
    cond = tf.logical_or(cond, pt.y(p) < top(r))
    cond = tf.logical_or(cond, pt.y(p) > bottom(r))
    return tf.logical_not(cond)


def clip(r, tb_range, lr_range):
    tb_min, tb_max = tf.split(tb_range, [1, 1])
    lr_min, lr_max = tf.split(lr_range, [1, 1])
    return create(
        pt.create(tf.clip_by_value(top(r), tb_min, tb_max), tf.clip_by_value(left(r), lr_min, lr_max)),
        pt.create(tf.clip_by_value(bottom(r), tb_min, tb_max), tf.clip_by_value(right(r), lr_min, lr_max)),
        r.dtype
    )


def clip_top_bottom(r, min, max):
    return create(
        pt.create(tf.clip_by_value(top(r), min, max), left(r)),
        pt.create(tf.clip_by_value(bottom(r), min, max), right(r)),
        r.dtype
    )

def clip_left_right(r, min, max):
    return create(
        pt.create(top(r), tf.clip_by_value(left(r), min, max)),
        pt.create(bottom(r), tf.clip_by_value(right(r), min, max)),
        r.dtype
    )


def nearest_point(r, p):
    x = tf.clip_by_value(pt.x(p), left(r), right(r))
    y = tf.clip_by_value(pt.y(p), top(r), bottom(r))
    return pt.create(y, x)


def convert_ratio_to_value(r, size, dtype=None):
    size = size - 1
    s = create(size, size)
    v = s * r
    return v if dtype is None else tf.cast(v, dtype)


def convert_value_to_ratio(r, size, dtype=None):
    size = size - 1
    s = create(size, size, dtype=tf.float32)
    v = r / s
    return v if dtype is None else tf.cast(v, dtype)