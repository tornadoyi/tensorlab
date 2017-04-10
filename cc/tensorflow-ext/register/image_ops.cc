#include "tensorflow/core/framework/op.h"
#include "register/common_shape_fns_ext.h"


REGISTER_OP("AssignImage")
        .Input("src: T")
        .Input("dst: float")
        .Input("rect: float")
        .Input("index: int32")
        .Output("output_image: float")
        .Attr("T: {uint8, int8, int16, int32, int64, half, float, double}")
        .SetShapeFn(shape_inference_ext::UnchangedInput1Shape);



REGISTER_OP("FlatColor")
        .Input("src: T")
        .Output("output_image: T")
        .Attr("T: {uint8, int8, int16, int32, int64, half, float, double}")
        .SetShapeFn([](shape_inference::InferenceContext* c) {
            auto input_shape = c->input(0);
            auto nb = c->Dim(input_shape, 0);
            auto nr = c->Dim(input_shape, 1);
            auto nc = c->Dim(input_shape, 2);
            auto nd = c->Dim(input_shape, 3);
            shape_inference::DimensionHandle new_row_size;
            TF_RETURN_IF_ERROR(c->Multiply(nr, nd, &new_row_size));
            auto output_shape = c->MakeShape({nb, new_row_size, nc});
            c->set_output(0, output_shape);
            return Status::OK();
        });



REGISTER_OP("Pyramid")
        .Input("images: T")
        .Input("scale: int32")
        .Output("pyramid_image: float")
        .Output("pyramid_rect: float")
        .Attr("T: {uint8, int8, int16, int32, int64, half, float, double}")
        .Attr("min_size: int = 5")
        .Attr("padding: int = 10")
        .SetShapeFn([](shape_inference::InferenceContext* c) {
            auto image_shape = c->input(0);
            auto nb = c->Dim(image_shape, 0);
            auto nc = c->Dim(image_shape, 2);
            auto nd = c->Dim(image_shape, 3);

            auto output_image_shape = c->MakeShape({nb, c->UnknownDim(), nc,  nd});
            auto rect_shape = c->MakeShape({c->UnknownDim(), c->MakeDim(4)});

            c->set_output(0, output_image_shape);
            c->set_output(1, rect_shape);
            return Status::OK();
        });




REGISTER_OP("PyramidApply")
        .Input("src: T")
        .Input("size: int32")
        .Input("rect: float")
        .Output("output_image: float")
        .Attr("T: {uint8, int8, int16, int32, int64, half, float, double}")
        .SetShapeFn([](shape_inference::InferenceContext* c) {
            shape_inference::ShapeHandle size;
            TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 4, &size));
            TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &size));

            auto image_shape = c->input(0);
            auto nb = c->Dim(image_shape, 0);
            auto nd = c->Dim(image_shape, 3);

            const auto size_tensor = c->input_tensor(1);
            shape_inference::DimensionHandle width, height;
            if (size_tensor == nullptr) {
                height = c->UnknownDim();
                width = c->UnknownDim();
            }
            else
            {
                auto vec = size_tensor->vec<int32>();
                height = c->MakeDim(vec(0));
                width = c->MakeDim(vec(1));
            }

            auto output_shape = c->MakeShape({nb, height, width, nd});
            c->set_output(0, output_shape);
            return Status::OK();
        });



REGISTER_OP("PyramidPlan")
        .Input("size: T")
        .Input("scale: T")
        .Output("output_size: T")
        .Output("rect: float")
        .Attr("min_size: int = 5")
        .Attr("padding: int = 10")
        .Attr("T: {int32, int64}")
        .SetShapeFn([](shape_inference::InferenceContext* c) {
            auto size_shape = c->input(0);
            auto nb = c->Dim(size_shape, 0);

            auto rect_shape = c->MakeShape({c->UnknownDim(), c->MakeDim(4)});

            c->set_output(0, c->input(0));
            c->set_output(1, rect_shape);
            return Status::OK();
        });



REGISTER_OP("PointToResizeSpace")
        .Input("point: T")
        .Input("scale: float")
        .Input("index: int32")
        .Output("output_image: T")
        .Attr("T: {uint8, int8, int16, int32, int64, half, float, double}")
        .SetShapeFn(shape_inference_ext::UnchangedInput0Shape);



REGISTER_OP("Test")
        .Input("src: T")
        .Input("dst: T")
        .Output("out1: T")
        .Output("out2: T")
        .Attr("T: {uint8, int8, int16, int32, int64, half, float, double}");