#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/resize_bilinear_op.h"
#include "tensorflow/core/kernels/image_resizer_state.h"
#include "image_tool.h"
#include "pyramid_op.h"

using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;


// reference from tensorflow/core/kernels/image_resizer_state.h
// modify input shape as argument
struct PyramidImageResizerState {
    explicit PyramidImageResizerState(bool align_corners)
            : align_corners_(align_corners) {}

    // ValidateAndCalculateOutputSize checks the bounds on the input tensors
    // and requested size, sets up some of the resizing state such as the
    // height_scale and width_scale, and calculates the output size.
    // If any of these operations fails, it sets an error status in
    // the context, which the caller must check.
    void ValidateAndCalculateOutputSize(OpKernelContext* context,
                                        const Tensor& input, const TensorShape& resize_shape) {
        OP_REQUIRES(context, input.dims() == 4,
                    errors::InvalidArgument("input must be 4-dimensional",
                                            input.shape().DebugString()));
        /*
        const Tensor& shape_t = context->input(1);
        OP_REQUIRES(context, shape_t.dims() == 1,
                    errors::InvalidArgument("shape_t must be 1-dimensional",
                                            shape_t.shape().DebugString()));
        OP_REQUIRES(context, shape_t.NumElements() == 2,
                    errors::InvalidArgument("shape_t must have two elements",
                                            shape_t.shape().DebugString()));
        */

        //auto Svec = shape_t.vec<int32>();
        batch_size = input.dim_size(0);
        //out_height = internal::SubtleMustCopy(Svec(0));
        //out_width = internal::SubtleMustCopy(Svec(1));
        out_height = resize_shape.dim_size(0);
        out_width = resize_shape.dim_size(1);

        OP_REQUIRES(
                context,
                FastBoundsCheck(input.dim_size(1), std::numeric_limits<int32>::max()) &&
                FastBoundsCheck(input.dim_size(2),
                                std::numeric_limits<int32>::max()),
                errors::InvalidArgument("input sizes must be between 0 and max int32"));

        in_height = static_cast<int32>(input.dim_size(1));
        in_width = static_cast<int32>(input.dim_size(2));
        channels = input.dim_size(3);
        OP_REQUIRES(context, out_height > 0 && out_width > 0,
                    errors::InvalidArgument("output dimensions must be positive"));
        OP_REQUIRES(
                context, channels > 0,
                errors::InvalidArgument("image must have at least one channel"));
        OP_REQUIRES(
                context, input.dim_size(1) > 0 && input.dim_size(2) > 0,
                errors::InvalidArgument("input image must be of non-zero size"));
        height_scale = CalculateResizeScale(in_height, out_height, align_corners_);
        width_scale = CalculateResizeScale(in_width, out_width, align_corners_);

        // Guard against overflows
        OP_REQUIRES(context,
                    ceilf((out_height - 1) * height_scale) <=
                    static_cast<float>(std::numeric_limits<int64>::max()),
                    errors::InvalidArgument(
                            "input image height scale would cause an overflow"));
        OP_REQUIRES(
                context,
                ceilf((out_width - 1) * width_scale) <= static_cast<float>(INT_MAX),
                errors::InvalidArgument(
                        "input image width scale would cause an overflow"));
    }

    // Calculates all the required variables, and allocates the output.
    template <typename T>
    void ValidateAndCreateOutput(OpKernelContext* context, const Tensor& input, const TensorShape& resize_shape, Tensor& output)
    {
        ValidateAndCalculateOutputSize(context, input, resize_shape);
        if (!context->status().ok()) return;
        /*
        OP_REQUIRES_OK(context, context->allocate_output(
                0, TensorShape({input.dim_size(0), out_height,
                                out_width, input.dim_size(3)}),
                &output));
        */

        TensorShape shape({input.dim_size(0), out_height, out_width, input.dim_size(3)});

        OP_REQUIRES_OK(context, context->allocate_temp(
                DataTypeToEnum<float>::v(), shape,
                &output));
    }

    int64 batch_size;
    int64 out_height;
    int64 out_width;
    int64 in_height;
    int64 in_width;
    int64 channels;
    float height_scale;
    float width_scale;

private:
    bool align_corners_;
};


enum E_RESIZE_METHOD{BILINEAR = 0, NEAREST_NEIGHBOR, BICUBIC, AREA};

template <typename Device, typename T>
class PyramidOp : public OpKernel
{
public:
    explicit PyramidOp(OpKernelConstruction* context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("align_corners", &align_corners_));
        OP_REQUIRES_OK(context, context->GetAttr("scale", &scale_));
        OP_REQUIRES_OK(context, context->GetAttr("min_size", &min_size_));
        OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));

        string s_method;
        OP_REQUIRES_OK(context, context->GetAttr("method", &s_method));
        std::transform(s_method.begin(), s_method.end(), s_method.begin(), ::toupper);
        if(s_method == "NEAREST_NEIGHBOR") method_ = NEAREST_NEIGHBOR;
        else if(s_method == "BICUBIC") method_ = BICUBIC;
        else if(s_method == "AREA") method_ = AREA;
        else{method_ = BILINEAR;}

    }

    void Compute(OpKernelContext* context) override
    {
        const Tensor& input_tensor = context->input(0);

        Tensor input_tensor_4d;
        CHECK(input_tensor_4d.CopyFrom(input_tensor, TensorShape({1, _nr(input_tensor), _nc(input_tensor), _nd(input_tensor)})));

        std::vector<Tensor> pyramid;
        pyramid.push_back(std::move(input_tensor_4d));
        while(true)
        {
            auto& input = pyramid[pyramid.size()-1];
            auto w = (int64)((scale_- 1) * input.dim_size(1) / scale_ + 0.5);
            auto h = (int64)((scale_- 1) * input.dim_size(2) / scale_ + 0.5);

            if(w < min_size_ || h < min_size_) break;

            Tensor output;
            ResizeImage(context, input, TensorShape({h, w}), output);
            pyramid.push_back(std::move(output));
        }


        int64 total_height = 0;
        for (auto&& t : pyramid)
            total_height += t.dim_size(2) + padding_;

        total_height -= padding_ * 2; // don't add unnecessary padding to the very right side.
        int64 height = 0;
        int64 prev_width = 0;
        for (auto&& t : pyramid)
        {
            // Figure out how far we go on the first column.  We go until the next image can
            // fit next to the previous one, which means we can double back for the second
            // column of images.
            if (t.dim_size(1) <= _nc(input_tensor)-prev_width-(int64)padding_ &&
                (height-_nr(input_tensor))*2 >= (total_height-_nr(input_tensor)))
            {
                break;
            }
            height += _nr(t) + (int64)padding_;
            prev_width = _nc(t);
        }
        height -= padding_; // don't add unnecessary padding to the very right side.

        Tensor* ouput_tensor;
        OP_REQUIRES_OK(context, context->allocate_output(
                0, TensorShape({height, _nc(input_tensor)}),
                &ouput_tensor));


        int64 y = 0;
        size_t i = 0;
        typename TTypes<float, 3>::Tensor dst = ouput_tensor->tensor<float, 3>();
        while(y < height)
        {
            //rectangle rect = translate_rect(get_rect(pyramid[i]),point(0,y));
            //DLIB_ASSERT(get_rect(out_img).contains(rect));
            //rects.push_back(rect);
            //auto si = sub_image(out_img, rect);
            //assign_image(si, pyramid[i]);
            const Tensor& input = pyramid[i];
            typename TTypes<T, 3>::ConstTensor src = input.tensor<T, 3>();
            //kernel::AssignImage<Device, T, float> m;
            //m(context->eigen_device<Device>(), src, dst, 0, (int)y);
            //y += _nr(input) + padding_;
            //++i;
        }



        //CHECK(ouput_tensor->CopyFrom(t, t.shape()));


    }

    void ResizeImage(OpKernelContext* context, const Tensor& input, const TensorShape& resize_shape, Tensor& output)
    {
        PyramidImageResizerState st(align_corners_);
        st.ValidateAndCreateOutput<T>(context, input, resize_shape, output);

        if (!context->status().ok()) return;

        // Return if the output is empty.
        if (output.NumElements() == 0) return;

        typename TTypes<T, 4>::ConstTensor image_data = input.tensor<T, 4>();
        typename TTypes<float, 4>::Tensor output_data = output.tensor<float, 4>();

        functor::ResizeBilinear<Device, T>()(context->eigen_device<Device>(),
                                             image_data, st.height_scale,
                                             st.width_scale, output_data);
    }


    void PyramidImages(OpKernelContext* context, const std::vector<Tensor>& images, int left_image_count,  Tensor& output)
    {

    }


    bool align_corners_;
    E_RESIZE_METHOD method_;
    int scale_;
    int padding_;
    float min_size_;
};


namespace kernel {
    template <typename T>
    struct PyramidImages<CPUDevice, T> {
        void operator()(const CPUDevice& d, typename TTypes<T, 4>::ConstTensor images,
                        typename TTypes<float, 4>::Tensor output)
        {

        }
    };
}  // namespace functor



#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("Pyramid")      \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T"), \
                          PyramidOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL





REGISTER_OP("Pyramid")
        .Input("images: T")
        .Input("scale: int32")
        .Output("resized_images: float")
        .Attr("T: {uint8, int8, int16, int32, int64, half, float, double}")
        .Attr("min_size: float = 5.0")
        .Attr("padding: int = 10")
        .Attr("method: string = 'BILINEAR'")
        .Attr("align_corners: bool = false");
        //.SetShapeFn(ResizeShapeFn)




