//
// Created by Jason on 17/03/2017.
//

#ifndef TENSORLAB_BIND_GEOMETRY_H
#define TENSORLAB_BIND_GEOMETRY_H

#include "bind_python_utils.h"
#include "geometry/geometry.h"


namespace wrap_rectangle
{

#define DECL_NS_RECTANGLE_DETAIL(cls_rect, cls_point, type) \
    namespace wrap_##cls_rect  \
    { \
        DECL_GETER_SETER(cls_rect, left); \
        DECL_GETER_SETER(cls_rect, top); \
        DECL_GETER_SETER(cls_rect, right); \
        DECL_GETER_SETER(cls_rect, bottom); \
        DECL_WRAP(cls_rect, width) \
        DECL_WRAP(cls_rect, height) \
        DECL_WRAP(cls_rect, area) \
        DECL_WRAP(cls_rect, empty) \
        DECL_WRAP(cls_rect, center) \
        DECL_WRAP_CORE_2(cls_rect, contains, contains_xy, type, type) \
        DECL_WRAP_CORE_1(cls_rect, contains, contains_point, const cls_point&) \
        DECL_WRAP_CORE_1(cls_rect, contains, contains_rect, const cls_rect&) \
        DECL_WRAP_1(cls_rect, intersect, const cls_rect&) \
        DECL_WRAP_STR(cls_rect) \
        cls_rect wrap_create_with_center(cls_point a, type c, type d){return cls_rect::create_with_center(a,c,d);} \
        cls_rect wrap_create_with_tlwh(cls_point a, type c, type d){return cls_rect::create_with_tlwh(a,c,d);} \
        void bind() \
        { \
            class_<cls_rect>(#cls_rect, "This object represents a rectangular area of an image.") \
                    .def(init<type,type,type,type>( (arg("left"),arg("top"),arg("right"),arg("bottom")) )) \
                    .add_property("left",   REF_GETER_SETER(left)) \
                    .add_property("top",    REF_GETER_SETER(top)) \
                    .add_property("right",  REF_GETER_SETER(right)) \
                    .add_property("bottom", REF_GETER_SETER(bottom)) \
                    .add_property("width", REF_WRAP(width)) \
                    .add_property("height", REF_WRAP(height)) \
                    .add_property("area", REF_WRAP(area)) \
                    .add_property("empty", REF_WRAP(empty)) \
                    .add_property("center", REF_WRAP(center)) \
                    .def("contains", REF_WRAP(contains_xy), arg("x"), arg("y")) \
                    .def("contains", REF_WRAP(contains_point), arg("point")) \
                    .def("contains", REF_WRAP(contains_rect), arg("rect")) \
                    .def("intersect", REF_WRAP(intersect), arg("rect")) \
                    .def("create_with_center", REF_WRAP(create_with_center), arg("center"), arg("width"), arg("height")) \
                    .def("create_with_tlwh", REF_WRAP(create_with_tlwh), arg("center"), arg("width"), arg("height")) \
                    .def("__str__", REF_WRAP_STR) \
                    .def("__repr__", REF_WRAP_STR) \
                    .def(self == self) \
                    .def(self != self) \
                    .staticmethod("create_with_center") \
                    .staticmethod("create_with_tlwh") \
                    ; \
        } \
    }

#define DECL_NS_RECTANGLE(type, suffix) \
    DECL_NS_RECTANGLE_DETAIL(Rectangle##suffix, Point2##suffix, type)


#define CALL_RECTANGLE_BIND(type, suffix) \
    wrap_Rectangle##suffix::bind(); \


    using boost::python::arg;

    DEF_REAL_NUMBER_TYPE_SUFFIX(DECL_NS_RECTANGLE)

    void bind()
    {
        DEF_REAL_NUMBER_TYPE_SUFFIX(CALL_RECTANGLE_BIND)
    }

}


namespace wrap_point
{

#define DECL_NS_POINT_2D_DETAIL(cls, type) \
    namespace wrap_##cls  \
    { \
        DECL_OPERATOR(cls, add, +) \
        DECL_OPERATOR(cls, sub, -) \
        DECL_WRAP_STR(cls); \
        DECL_GET_ITEM(cls, get_item, type) \
        void bind() \
        { \
            class_<cls>(#cls, "This object represents a point.") \
                    .def(init<type, type>( (arg("x"), arg("y")))) \
                    .def("__str__", REF_WRAP_STR) \
                    .def("__repr__", REF_WRAP_STR) \
                    .def("__add__", REF_WRAP(add), arg("other")) \
                    .def("__sub__", REF_WRAP(sub), arg("other")) \
                    .def("__getitem__", REF_WRAP(get_item)) \
                    ; \
        } \
    }

#define DECL_NS_POINT_3D_DETAIL(cls, type) \
    namespace wrap_##cls  \
    { \
        DECL_OPERATOR(cls, add, +) \
        DECL_OPERATOR(cls, sub, -) \
        DECL_WRAP_STR(cls); \
        DECL_GET_ITEM(cls, get_item, type) \
        void bind() \
        { \
            class_<cls>(#cls, "This object represents a point.") \
                    .def(init<type, type, type>( (arg("x"), arg("y"), arg("z")))) \
                    .def("__str__", REF_WRAP_STR) \
                    .def("__repr__", REF_WRAP_STR) \
                    .def("__add__", REF_WRAP(add), arg("other")) \
                    .def("__sub__", REF_WRAP(sub), arg("other")) \
                    .def("__getitem__", REF_WRAP(get_item)) \
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
