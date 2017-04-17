#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/resize_bilinear_op.h"
#include "tensorflow/core/kernels/image_resizer_state.h"
#include "image_tool.h"
#include "assign_image_op.h"
#include "pyramid_plan_op.h"
#include "pyramid_op.h"


using namespace tensorflow;
typedef Eigen::ThreadPoolDevice CPUDevice;


template <typename Device, typename T>
class PyramidOp : public OpKernel
{
public:
    explicit PyramidOp(OpKernelConstruction* context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("min_size", &min_size_));
        OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
    }

    void Compute(OpKernelContext* context) override
    {
        const Tensor& input_tensor = context->input(0);
        const Tensor& scale_tensor = context->input(1);


        OP_REQUIRES(context, input_tensor.dims() == 4,
                    errors::InvalidArgument("image must be 4-dimensional",
                                            input_tensor.shape().DebugString()));


        OP_REQUIRES(context, TensorShapeUtils::IsScalar(scale_tensor.shape()),
                    errors::InvalidArgument("scale must be a scalar",
                                            scale_tensor.shape().DebugString()));

        auto scale_ = scale_tensor.flat<int32>()(0);

        int32 output_height, output_width;
        std::vector<std::tuple<float, float, float, float>> output_rects;
        MakePyramidPlan<int32, 0>()(
                (int32)_nr(input_tensor),
                (int32)_nc(input_tensor),
                scale_,
                min_size_,
                padding_,
                output_height,
                output_width,
                output_rects
        );


        Tensor* output_tensor;
        OP_REQUIRES_OK(context, context->allocate_output(
                0, TensorShape({_nb(input_tensor), output_height, output_width, _nd(input_tensor)}),
                &output_tensor));

        Tensor* rects_tensor;
        OP_REQUIRES_OK(context, context->allocate_output(
                1, TensorShape({(int32)output_rects.size(), 4}),
                &rects_tensor));

        Tensor indexes_tensor;
        OP_REQUIRES_OK(context, context->allocate_temp(
                DataTypeToEnum<int32>::v(),
                TensorShape({_nb(input_tensor) * (int32)output_rects.size(), 3}),
                &indexes_tensor));


        // generate rects and indexes
        auto ptr_rects = rects_tensor->tensor<float, 2>();
        auto ptr_index = indexes_tensor.tensor<int32, 2>();
        auto batch = _nb(input_tensor);
        for(size_t i=0; i<output_rects.size(); ++i)
        {
            auto& rect = output_rects[i];
            ptr_rects(i, 0) = std::get<0>(rect);
            ptr_rects(i, 1) = std::get<1>(rect);
            ptr_rects(i, 2) = std::get<2>(rect);
            ptr_rects(i, 3) = std::get<3>(rect);

            //LOG(INFO) << std::get<0>(rect) << " "<< std::get<1>(rect) << " " << std::get<2>(rect) << " "<< std::get<3>(rect) << " ";

            auto st = i * batch;
            for(int64 j = 0; j<batch; ++j)
            {
                auto index = st + j;
                ptr_index(index, 0) = j;    // src
                ptr_index(index, 1) = j;    // dst
                ptr_index(index, 2) = i;    // rect
            }
        }

        // set output to zeros IMPORTANT !!!!
        output_tensor->tensor<float, 4>().setZero();

        kernel::AssignImage<Device, T>()(
                 context->eigen_device<Device>(),
                 input_tensor.tensor<T, 4>(),
                 ((const Tensor*)rects_tensor)->tensor<float, 2>(),
                 ((const Tensor)indexes_tensor).tensor<int32, 2>(),
                 output_tensor->tensor<float, 4>()
        );


    }

    int32 padding_;
    int32 min_size_;
};




#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("Pyramid")      \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T"), \
                          PyramidOp<CPUDevice, T>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL









