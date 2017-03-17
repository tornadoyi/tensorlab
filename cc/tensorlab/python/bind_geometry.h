//
// Created by Jason on 17/03/2017.
//

#ifndef TENSORLAB_BIND_GEOMETRY_H
#define TENSORLAB_BIND_GEOMETRY_H

#include "bind_python_utils.h"
#include "geometry/geometry.h"


namespace wrap_rectangle
{
    using boost::python::arg;

    DECL_GETER_SETER(Rectangle, left);
    DECL_GETER_SETER(Rectangle, top);
    DECL_GETER_SETER(Rectangle, right);
    DECL_GETER_SETER(Rectangle, bottom);
    DECL_WRAP(Rectangle, width)
    DECL_WRAP(Rectangle, height)
    DECL_WRAP(Rectangle, area)
    DECL_WRAP(Rectangle, empty)
    DECL_WRAP(Rectangle, center)
    DECL_WRAP_CORE_2(Rectangle, contains, contains_xy, long, long)
    DECL_WRAP_CORE_1(Rectangle, contains, contains_point, const Point2l&)
    DECL_WRAP_CORE_1(Rectangle, contains, contains_rect, const Rectangle&)
    DECL_WRAP_1(Rectangle, intersect, const Rectangle&)
    DECL_WRAP_STR(Rectangle)

    void bind()
    {
        typedef Rectangle type;

        class_<type>("Rectangle", "This object represents a rectangular area of an image.")
                .def(init<long,long,long,long>( (arg("left"),arg("top"),arg("right"),arg("bottom")) ))
                .add_property("left",   REF_GETER_SETER(left))
                .add_property("top",    REF_GETER_SETER(top))
                .add_property("right",  REF_GETER_SETER(right))
                .add_property("bottom", REF_GETER_SETER(bottom))
                .def("width", REF_WRAP(width))
                .def("height", REF_WRAP(height))
                .def("area", REF_WRAP(area))
                .def("empty", REF_WRAP(empty))
                .def("center", REF_WRAP(center))
                .def("contains", REF_WRAP(contains_xy), arg("x"), arg("y"))
                .def("contains", REF_WRAP(contains_point), arg("point"))
                .def("contains", REF_WRAP(contains_rect), arg("rect"))
                .def("intersect", REF_WRAP(intersect), arg("rect"))
                .def("__str__", REF_WRAP_STR)
                .def("__repr__", REF_WRAP_STR)
                .def(self == self)
                .def(self != self)
            ;

    }
}


namespace wrap_point
{

#define DECL_NS_POINT_2D_DETAIL(cls, type) \
    namespace wrap_##cls  \
    { \
        DECL_GETER_SETER(cls, x); \
        DECL_GETER_SETER(cls, y); \
        DECL_WRAP_STR(cls); \
        void bind() \
        { \
            class_<cls>(#cls, "This object represents a point.") \
                    .def(init<type, type>( (arg("x"), arg("y")))) \
                    .add_property("x",   REF_GETER_SETER(x)) \
                    .add_property("y",   REF_GETER_SETER(y)) \
                    .def("__str__", REF_WRAP_STR) \
                    ; \
        } \
    }

#define DECL_NS_POINT_3D_DETAIL(cls, type) \
    namespace wrap_##cls  \
    { \
        DECL_GETER_SETER(cls, x); \
        DECL_GETER_SETER(cls, y); \
        DECL_GETER_SETER(cls, z); \
        DECL_WRAP_STR(cls); \
        void bind() \
        { \
            class_<cls>(#cls, "This object represents a point.") \
                    .def(init<type, type, type>( (arg("x"), arg("y"), arg("z")))) \
                    .add_property("x",   REF_GETER_SETER(x)) \
                    .add_property("y",   REF_GETER_SETER(y)) \
                    .add_property("z",   REF_GETER_SETER(z)) \
                    .def("__str__", REF_WRAP_STR) \
                    ; \
        } \
    }

#define DECL_NS_POINT(type, suffix) \
    DECL_NS_POINT_2D_DETAIL(Point2##suffix, type) \
    DECL_NS_POINT_3D_DETAIL(Point3##suffix, type)


#define CALL_POINT_BIND(type, suffix) \
    wrap_Point2##suffix::bind(); \
    wrap_Point3##suffix::bind();



    using boost::python::arg;

    DEF_ALL_TYPE_SUFFIX(DECL_NS_POINT)

    void bind()
    {
        DEF_ALL_TYPE_SUFFIX(CALL_POINT_BIND)
    }

}


void bind_geometry()
{
    wrap_rectangle::bind();
    wrap_point::bind();
}

#endif //PROJECT_BIND_GEOMETRY_H
