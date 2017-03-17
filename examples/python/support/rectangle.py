# Rectangle data must be 1-dimensons of tuple or array with 4 elements
# example (top, left, width, height)

def __call__(top, left, width, height): return (top, left, width, height)

def _top(rect): return rect[0]
def _left(rect): return rect[1]
def _width(rect): return rect[2]
def _height(rect): return rect[3]
def _right(rect): return _left

def center(rect):
    _left(rect) + _ri
    temp(rect.left() + rect.right() + 1,
         rect.top() + rect.bottom() + 1);

    if (temp.x() < 0)
        temp.x() -= 1;

    if (temp.y() < 0)
        temp.y() -= 1;

    return temp / 2;