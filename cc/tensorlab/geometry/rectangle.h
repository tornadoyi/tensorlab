//
// Created by Jason on 16/03/2017.
//

#ifndef TENSORLAB_RECTANGLE_H
#define TENSORLAB_RECTANGLE_H

#include "Eigen/Dense"
#include "base/base.h"
#include "point.h"


using namespace Eigen;
using namespace std;

template <typename T>
class Rectangle
{
    /*!
        INITIAL VALUE
            The initial value of this object is defined by its constructor.

        CONVENTION
            left() == l
            top() == t
            right() == r
            bottom() == b
    !*/

public:

    Rectangle (
            T l_,
            T t_,
            T r_,
            T b_
    ) :
            l(l_),
            t(t_),
            r(r_),
            b(b_)
    {}

    Rectangle (
            T w,
            T h
    ) :
            l(0),
            t(0),
            r(static_cast<T>(w)-1),
            b(static_cast<T>(h)-1)
    {
        TL_ASSERT((w > 0 && h > 0) || (w == 0 && h == 0),
                    "\tRectangle(width,height)"
                            << "\n\twidth and height must be > 0 or both == 0"
                            << "\n\twidth:  " << w
                            << "\n\theight: " << h
                            << "\n\tthis: " << this
        );
    }

    Rectangle (
            const Point2D<T>& p
    ) :
            l(p.x()),
            t(p.y()),
            r(p.x()),
            b(p.y())
    {
    }

    Rectangle (
            const Point2D<T>& p1,
            const Point2D<T>& p2
    )
    {
        *this = Rectangle(p1) + Rectangle(p2);
    }

    static Rectangle create_with_center(Point2D<T> p, T width, T height)
    {
        return create_with_center(p[0], p[1], width, height);
    }

    static Rectangle create_with_center(T centerX, T centerY, T width, T height)
    {
        auto left = (centerX - width) / (T)2;
        auto top = (centerY - height) / (T)2;
        auto right = left + width - 1;
        auto bottom = top + height - 1;
        return Rectangle(left, top, right, bottom);
    }

    static Rectangle create_with_tlwh(Point2D<T> p, T widh, T height)
    {
        return create_with_tlwh(p[0], p[1], widh, height);
    }

    static Rectangle create_with_tlwh(T top, T left, T width, T height)
    {
        return Rectangle(left, top, left + width - 1, top + height - 1);
    }

    /*
    template <typename T>
    Rectangle (
            const vector<T,2>& p1,
            const vector<T,2>& p2
    )
    {
        *this = Rectangle(p1) + Rectangle(p2);
    }
     */

    Rectangle (
    ) :
            l(0),
            t(0),
            r(-1),
            b(-1)
    {}

    void set_top(T, T){}

    T top (
    ) const { return t; }

    T& top (
    ) { return t; }


    T left (
    ) const { return l; }

    T& left (
    ) { return l; }


    T right (
    ) const { return r; }

    T& right (
    ) { return r; }


    T bottom (
    ) const { return b; }

    T& bottom (
    ) { return b; }


    const Point2D<T> tl_corner (
    ) const { return Point2D<T>(left(), top()); }

    const Point2D<T> bl_corner (
    ) const { return Point2D<T>(left(), bottom()); }

    const Point2D<T> tr_corner (
    ) const { return Point2D<T>(right(), top()); }

    const Point2D<T> br_corner (
    ) const { return Point2D<T>(right(), bottom()); }

    T width (
    ) const
    {
        if (empty())
            return 0;
        else
            return r - l + 1;
    }

    T height (
    ) const
    {
        if (empty())
            return 0;
        else
            return b - t + 1;
    }

    T area (
    ) const
    {
        return width()*height();
    }

    Point2D<T> center (
    )
    {
        Point2D<T> temp(left() + right() + 1,
                   top() + bottom() + 1);

        if (temp.x() < 0)
            temp.x() -= 1;

        if (temp.y() < 0)
            temp.y() -= 1;

        temp /= 2;
        return temp;
    }

    bool empty (
    ) const { return (t > b || l > r); }

    Rectangle operator + (
            const Rectangle& rhs
    ) const
    {
        if (rhs.empty())
            return *this;
        else if (empty())
            return rhs;

        return Rectangle (
                std::min(l,rhs.l),
                std::min(t,rhs.t),
                std::max(r,rhs.r),
                std::max(b,rhs.b)
        );
    }

    Rectangle intersect (
            const Rectangle& rhs
    ) const
    {
        return Rectangle (
                std::max(l,rhs.l),
                std::max(t,rhs.t),
                std::min(r,rhs.r),
                std::min(b,rhs.b)
        );
    }

    bool contains (
            const Point2D<T>& p
    ) const
    {
        if (p.x() < l || p.x() > r || p.y() < t || p.y() > b)
            return false;
        return true;
    }

    bool contains (
            T x,
            T y
    ) const
    {
        if (x < l || x > r || y < t || y > b)
            return false;
        return true;
    }

    bool contains (
            const Rectangle& rect
    ) const
    {
        return (rect + *this == *this);
    }

    Rectangle& operator+= (
            const Point2D<T>& p
    )
    {
        *this = *this + Rectangle(p);
        return *this;
    }

    Rectangle& operator+= (
            const Rectangle& rect
    )
    {
        *this = *this + rect;
        return *this;
    }

    bool operator== (
            const Rectangle& rect
    ) const
    {
        return (l == rect.l) && (t == rect.t) && (r == rect.r) && (b == rect.b);
    }

    bool operator!= (
            const Rectangle& rect
    ) const
    {
        return !(*this == rect);
    }

    inline bool operator< (const Rectangle& b) const
    {
        if      (left() < b.left()) return true;
        else if (left() > b.left()) return false;
        else if (top() < b.top()) return true;
        else if (top() > b.top()) return false;
        else if (right() < b.right()) return true;
        else if (right() > b.right()) return false;
        else if (bottom() < b.bottom()) return true;
        else if (bottom() > b.bottom()) return false;
        else                    return false;
    }


private:
    T l;
    T t;
    T r;
    T b;
};

template <typename T>
inline std::ostream& operator<< (
        std::ostream& out,
        const Rectangle<T>& item
)
{
    out << "[(" << item.left() << ", " << item.top() << ") (" << item.right() << ", " << item.bottom() << ")]";
    return out;
}


#define DEF_ALL_CLASS(Type, Suffix) \
    typedef Rectangle<Type> Rectangle##Suffix;


DEF_REAL_NUMBER_TYPE_SUFFIX(DEF_ALL_CLASS);
#undef DEF_ALL_CLASS


#endif //PROJECT_RECTANGLE_H
