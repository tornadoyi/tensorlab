//
// Created by Jason on 16/03/2017.
//

#ifndef TENSORLAB_POINT_H
#define TENSORLAB_POINT_H

#include "base/base.h"
#include "Eigen/Dense"

using namespace Eigen;
using namespace std;

template <typename Type, int Size>
class Point : public Matrix<Type, Size, 1>
{
public:
    Point(){}
};

template <typename T>
class Point2D : public Point<T, 2>
{
public:
    Point2D(){}
    Point2D(T x, T y){(*this)(0) = x; (*this)(1) = y;}
    T x() const{ return (*this)(0);}
    T& x(){ return (*this)(0);}

    T y() const{ return (*this)(1);}
    T& y(){ return (*this)(1);}
};

template <typename T>
class Point3D : public Point<T, 3>
{
public:
    Point3D(){}
    Point3D(T x, T y, T z){(*this)(0) = x; (*this)(1) = y; (*this)(2) = z;}
    T x() const{ return (*this)(0);}
    T& x(){ return (*this)(0);}

    T y() const{ return (*this)(1);}
    T& y(){ return (*this)(1);}

    T z() const{ return (*this)(1);}
    T& z(){ return (*this)(2);}
};

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

#define DEF_ALL_CLASS(Type, Suffix) \
    typedef Point2D<Type> Point2##Suffix; \
    typedef Point3D<Type> Point3##Suffix;


DEF_ALL_TYPE_SUFFIX(DEF_ALL_CLASS);
#undef DEF_ALL_CLASS

#endif //PROJECT_POINT_H
