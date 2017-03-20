from point import Point

class Rectangle():
    def __init__(self, left, top, right, bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    @staticmethod
    def create_with_tlwh(top, left, width, height): return Rectangle(left, top, left+width-1, top+height-1)

    @staticmethod
    def create_with_center(centerX, centerY,  width, height):
        left = (centerX - width) / 2
        top = (centerY - height) / 2
        right = left + width - 1
        bottom = top + height - 1
        return Rectangle(left, top, right, bottom)

    @property
    def width(self): return self.right - self.left + 1

    @property
    def height(self): return self.bottom - self.top + 1

    @property
    def area(self): return self.width * self.height

    @property
    def center(self):
        temp = Point(self.left + self.right + 1, self.top + self.bottom + 1);

        if temp.x < 0:
            temp.x = temp.x - 1

        if temp.y < 0:
            temp.y = temp.y - 1

        temp /= 2;
        return temp;
