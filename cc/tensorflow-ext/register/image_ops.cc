#include "tensorflow/core/framework/op.h"


REGISTER_OP("AssignImage")
        .Input("src: T")
        .Input("dst: float")
        .Input("rect: int32")
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
        .Attr("T: {uint8, int8, int16, int32, int64, half, float, double}")
        .Attr("min_size: int = 5")
        .Attr("padding: int = 10");
        //.SetShapeFn(ResizeShapeFn)


REGISTER_OP("PyramidApply")
        .Input("src: T")
        .Input("rect: int32")
        .Output("output_image: float")
        .Attr("T: {uint8, int8, int16, int32, int64, half, float, double}");


REGISTER_OP("PyramidPlan")
        .Input("size: T")
        .Input("scale: T")
        .Attr("min_size: int = 5")
        .Attr("padding: int = 10")
        .Output("output_image: T")
        .Attr("T: {int32, int64}");