#include "tensorflow/core/framework/op.h"


REGISTER_OP("AssignImage")
.Input("src: T")
.Input("dst: T")
.Input("location: int32")
.Output("output_image: T")
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
        .Attr("min_size: float = 5.0")
        .Attr("padding: int = 10")
        .Attr("method: string = 'BILINEAR'")
        .Attr("align_corners: bool = false");
        //.SetShapeFn(ResizeShapeFn)