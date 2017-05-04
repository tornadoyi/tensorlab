//
// Created by Jason on 16/03/2017.
//

#ifndef TENSORLAB_POINT_H
#define TENSORLAB_POINT_H

#include "base/base.h"
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

template <typename T, int Size> using Point = Matrix<T, Size, 1>;
template <typename T> using Point2D = Point<T, 2>;
template <typename T> using Point3D = Point<T, 3>;


template <typename Type, int Size>
inline std::ostream& operator<< (
        std::ostream& out,
        const Point<Type, Size>& item
)
{
    out << "(";
    for(auto i=0; i<Size; ++i)
    {
        out << item(i);
        if(i < Size - 1) out << ", ";
    }
    out << ")";
    return out;
}


template<typename T>
Point2D<T> operator+(const Point2D<T>& a, const Point2D<T>& b)
{
    Point2D<T> c;
    for(auto i=0; i<2; ++i)
    {
        c(i) = a(i) + b(i);
    }
    return c;
}




#define DECL_POINT_CLASS(Type, Suffix) \
typedef Point2D<Type> Point2##Suffix; \
typedef Point3D<Type> Point3##Suffix;


DEF_ALL_TYPE_SUFFIX(DECL_POINT_CLASS);
#undef DEF_ALL_CLASS


#endif //PROJECT_POINT_H
