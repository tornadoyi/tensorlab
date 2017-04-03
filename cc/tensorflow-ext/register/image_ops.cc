#include "tensorflow/core/framework/op.h"


REGISTER_OP("AssignImage")
        .Input("src: T")
        .Input("dst: float")
        .Input("rect: float")
        .Input("index: int32")
        .Output("output_image: float")
        .Attr("T: {uint8, int8, int16, int32, int64, half, float, double}");



REGISTER_OP("FlatColor")
        .Input("src: T")
        .Output("output_image: T")
        .Attr("T: {uint8, int8, int16, int32, int64, half, float, double}");



REGISTER_OP("Pyramid")
        .Input("images: T")
        .Input("scale: int32")
        .Output("pyramid_image: float")
        .Output("pyramid_rect: float")
        .Attr("T: {uint8, int8, int16, int32, int64, half, float, double}")
        .Attr("min_size: int = 5")
        .Attr("padding: int = 10");
        //.SetShapeFn(ResizeShapeFn)


REGISTER_OP("PyramidApply")
        .Input("src: T")
        .Input("size: int32")
        .Input("rect: float")
        .Output("output_image: float")
        .Attr("T: {uint8, int8, int16, int32, int64, half, float, double}");


REGISTER_OP("PyramidPlan")
        .Input("size: T")
        .Input("scale: T")
        .Output("output_size: T")
        .Output("rect: float")
        .Attr("min_size: int = 5")
        .Attr("padding: int = 10")
        .Attr("T: {int32, int64}");


REGISTER_OP("PointToResizeSpace")
        .Input("point: T")
        .Input("scale: float")
        .Input("index: int32")
        .Output("output_image: T")
        .Attr("T: {uint8, int8, int16, int32, int64, half, float, double}");

REGISTER_OP("Test")
        .Input("src: T")
        .Input("dst: T")
        .Output("out1: T")
        .Output("out2: T")
        .Attr("T: {uint8, int8, int16, int32, int64, half, float, double}");