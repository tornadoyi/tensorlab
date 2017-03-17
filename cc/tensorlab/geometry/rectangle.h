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
            long l_,
            long t_,
            long r_,
            long b_
    ) :
            l(l_),
            t(t_),
            r(r_),
            b(b_)
    {}

    Rectangle (
            unsigned long w,
            unsigned long h
    ) :
            l(0),
            t(0),
            r(static_cast<long>(w)-1),
            b(static_cast<long>(h)-1)
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
            const Point2l& p
    ) :
            l(p.x()),
            t(p.y()),
            r(p.x()),
            b(p.y())
    {
    }

    Rectangle (
            const Point2l& p1,
            const Point2l& p2
    )
    {
        *this = Rectangle(p1) + Rectangle(p2);
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

    void set_top(long, long){}

    long top (
    ) const { return t; }

    long& top (
    ) { return t; }


    long left (
    ) const { return l; }

    long& left (
    ) { return l; }


    long right (
    ) const { return r; }

    long& right (
    ) { return r; }


    long bottom (
    ) const { return b; }

    long& bottom (
    ) { return b; }


    const Point2l tl_corner (
    ) const { return Point2l(left(), top()); }

    const Point2l bl_corner (
    ) const { return Point2l(left(), bottom()); }

    const Point2l tr_corner (
    ) const { return Point2l(right(), top()); }

    const Point2l br_corner (
    ) const { return Point2l(right(), bottom()); }

    unsigned long width (
    ) const
    {
        if (empty())
            return 0;
        else
            return r - l + 1;
    }

    unsigned long height (
    ) const
    {
        if (empty())
            return 0;
        else
            return b - t + 1;
    }

    unsigned long area (
    ) const
    {
        return width()*height();
    }

    Point2l center (
    )
    {
        Point2l temp(left() + right() + 1,
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
            const Point2l& p
    ) const
    {
        if (p.x() < l || p.x() > r || p.y() < t || p.y() > b)
            return false;
        return true;
    }

    bool contains (
            long x,
            long y
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
            const Point2l& p
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
    long l;
    long t;
    long r;
    long b;
};

inline std::ostream& operator<< (
        std::ostream& out,
        const Rectangle& item
)
{
    out << "[(" << item.left() << ", " << item.top() << ") (" << item.right() << ", " << item.bottom() << ")]";
    return out;
}

#endif //PROJECT_RECTANGLE_H
