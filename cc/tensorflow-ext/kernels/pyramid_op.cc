#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/resize_bilinear_op.h"
#include "tensorflow/core/kernels/image_resizer_state.h"
#include "image_tool.h"
#include "assign_image_op.h"
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

        OP_REQUIRES(context, resize_shape.dims() == 2,
                    errors::InvalidArgument("resize_shape must be 1-dimensional",
                                            resize_shape.DebugString()));


        //auto Svec = shape_t.vec<int32>();
        batch_size = _nb(input);
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

        TensorShape shape({batch_size, out_height, out_width, channels});

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
        const Tensor& scale_tensor = context->input(1);
        auto scale_ = scale_tensor.flat<int32>()(0);

        OP_REQUIRES(context, input_tensor.dims() == 4,
                    errors::InvalidArgument("image must be 4-dimensional",
                                            input_tensor.shape().DebugString()));

        std::vector<Tensor> pyramid;
        do
        {
            if(pyramid.size() == 0)
            {
                Tensor output;
                ResizeImage<T>(context, input_tensor, TensorShape({_nr(input_tensor), _nc(input_tensor)}), output);
                pyramid.push_back(std::move(output));
                continue;
            }

            auto& input = pyramid[pyramid.size()-1];
            auto r = (int64)((scale_- 1) * _nr(input) / scale_ + 0.5);
            auto c = (int64)((scale_- 1) * _nc(input) / scale_ + 0.5);


            if(r < min_size_ || c < min_size_) break;

            Tensor output;
            ResizeImage<float>(context, input, TensorShape({r, c}), output);
            pyramid.push_back(std::move(output));

        }while (true);

        int64 total_height = 0;
        for (auto&& t : pyramid)
            total_height += _nr(t) + padding_;

        total_height -= padding_ * 2; // don't add unnecessary padding to the very right side.
        int64 height = 0;
        int64 prev_width = 0;
        for (auto&& t : pyramid)
        {
            // Figure out how far we go on the first column.  We go until the next image can
            // fit next to the previous one, which means we can double back for the second
            // column of images.
            if (_nc(t) <= _nc(input_tensor)-prev_width-(int64)padding_ &&
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
                0, TensorShape({_nb(input_tensor), height, _nc(input_tensor), _nd(input_tensor)}),
                &ouput_tensor));


        int64 y = 0;
        size_t i = 0;
        typename TTypes<float, 4>::Tensor dst = ouput_tensor->tensor<float, 4>();
        while(y < height)
        {
            const Tensor& input = pyramid[i];
            typename TTypes<float, 4>::ConstTensor src = input.tensor<float, 4>();
            kernel::AssignImage<Device, float, 4>()(context->eigen_device<Device>(), src, (int)y, 0, dst);

            y += _nr(input) + padding_;
            ++i;
        }
        y -= padding_;

        while (i < pyramid.size())
        {
            const Tensor& input = pyramid[i];
            auto br_x = _nc(dst) - 1;
            auto br_y = y - 1;
            auto tl_x = br_x - _nc(input);
            auto tl_y = br_y - _nr(input);

            typename TTypes<float, 4>::ConstTensor src = input.tensor<float, 4>();
            kernel::AssignImage<Device, float, 4>()(context->eigen_device<Device>(), src, tl_y, tl_x, dst);

            y -= _nr(input) + padding_;
            ++i;
        }

    }

    template <typename T_IMAGE>
    void ResizeImage(OpKernelContext* context, const Tensor& input, const TensorShape& resize_shape, Tensor& output)
    {
        PyramidImageResizerState st(align_corners_);
        st.ValidateAndCreateOutput<T_IMAGE>(context, input, resize_shape, output);

        if (!context->status().ok()) return;

        // Return if the output is empty.
        if (output.NumElements() == 0) return;

        typename TTypes<T_IMAGE, 4>::ConstTensor image_data = input.tensor<T_IMAGE, 4>();
        typename TTypes<float, 4>::Tensor output_data = output.tensor<float, 4>();

        functor::ResizeBilinear<Device, T_IMAGE>()(context->eigen_device<Device>(),
                                             image_data, st.height_scale,
                                             st.width_scale, output_data);
    }


    bool align_corners_;
    E_RESIZE_METHOD method_;
    int padding_;
    float min_size_;
};




#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("Pyramid")      \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T"), \
                          PyramidOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL









