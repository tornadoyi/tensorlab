//
// Created by Jason on 31/03/2017.
//


#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/resize_bilinear_op.h"
#include "tensorflow/core/kernels/image_resizer_state.h"
#include "image_tool.h"
#include "assign_image_op.h"
#include "pyramid_plan_op.h"

typedef Eigen::ThreadPoolDevice CPUDevice;

enum E_RESIZE_METHOD{BILINEAR = 0, NEAREST_NEIGHBOR, BICUBIC, AREA};

template <typename Device, typename T>
class PyramidPlanOp : public OpKernel
{
public:
    explicit PyramidPlanOp(OpKernelConstruction* context) : OpKernel(context)
    {
        OP_REQUIRES_OK(context, context->GetAttr("min_size", &min_size));
        OP_REQUIRES_OK(context, context->GetAttr("padding", &padding));
    }

    void Compute(OpKernelContext* context) override
    {
        const Tensor& size_tensor = context->input(0);
        const Tensor& scaletensor = context->input(1);


        OP_REQUIRES(context, size_tensor.dims() == 1,
                    errors::InvalidArgument("size must be 1-dimensional",
                                            size_tensor.shape().DebugString()));

        OP_REQUIRES(context, size_tensor.NumElements() == 2,
                    errors::InvalidArgument("size must be 2 elements",
                                            size_tensor.DebugString()));

        auto input_size = size_tensor.flat<T>();
        auto scale = scaletensor.flat<T>()(0);

        T input_r = input_size(0);
        T input_c = input_size(1);

        std::vector<std::tuple<T, T, T, T>> rects;
        T output_width, output_height;
        MakePyramidPlan<T>()(input_r, input_c, scale, min_size, padding, output_height, output_width, rects);
        //LOG(INFO) << output_height << " " << output_width;

        Tensor* output_tensor;
        OP_REQUIRES_OK(context, context->allocate_output(
                0,
                TensorShape({int64(rects.size()+1), 4}),
                &output_tensor));

        typename TTypes<T, 2>::Tensor output = output_tensor->tensor<T, 2>();
        T* ptr = output.data();
        ptr[0] = 0;
        ptr[1] = 0;
        ptr[2] = output_height;
        ptr[3] = output_width;
        ptr += 4;

        for(size_t i=0; i<rects.size(); ++i)
        {
            auto& rect = rects[i];
            ptr[0] = std::get<0>(rect);
            ptr[1] = std::get<1>(rect);
            ptr[2] = std::get<2>(rect);
            ptr[3] = std::get<3>(rect);
            ptr += 4;
        }
    }

    int padding;
    int min_size;
};


template <typename T>
struct MakePyramidPlan<T, 0>
{
    void operator()(
            T input_height,
            T intput_width,
            T scale,
            T min_size,
            T padding,
            T& output_height,
            T& output_width,
            std::vector<std::tuple<T, T, T, T>>& output_rects)
    {
        output_rects.clear();

        T r, c;

        T total_height = 0;
        r = input_height;
        c = intput_width;
        std::vector<std::pair<T, T>> pyramid;
        pyramid.push_back(std::pair<T, T>(r, c));
        do
        {
            r = (T)((scale- 1) * r / scale + 0.5);
            c = (T)((scale- 1) * c / scale + 0.5);
            if(r < min_size || c < min_size) break;

            pyramid.push_back(std::pair<T, T>(r, c));
            total_height += r + padding;

        }while (true);
        total_height -= padding * 2; // don't add unnecessary padding to the very right side.

        T height = 0;
        T prev_width = 0;

        for (auto&& t : pyramid)
        {
            r = t.first; c = t.second;
            if (c <= intput_width - prev_width - padding &&
                (height - input_height) * 2 >= (total_height - input_height))
            {
                break;
            }
            height += r + padding;
            prev_width = c;
        }

        height -= padding; // don't add unnecessary padding to the very right side.

        output_height = height;
        output_width = intput_width;


        T y = 0;
        size_t i = 0;
        while(y < height)
        {
            auto& size = pyramid[i];
            r = size.first; c = size.second;

            std::tuple<T, T, T, T> rect(y, 0, r, c);
            output_rects.push_back(std::move(rect));

            y += r + padding;
            ++i;
        }
        y -= padding;


        while (i < pyramid.size())
        {
            auto& size = pyramid[i];
            r = size.first; c = size.second;

            auto br_x = output_width - 1;
            auto br_y = y - 1;
            auto tl_x = br_x - c;
            auto tl_y = br_y - r;

            std::tuple<T, T, T, T> rect(tl_y, tl_x, r, c);
            output_rects.push_back(rect);
            y -= r + padding;
            ++i;
        }

    }
};


#define DEFINE_SPECS(T)                    \
    template struct MakePyramidPlan<T, 0>; \

TF_CALL_int32(DEFINE_SPECS);
TF_CALL_int64(DEFINE_SPECS);

#undef DEFINE_CPU_SPECS



#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("PyramidPlan")      \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T"), \
                          PyramidPlanOp<CPUDevice, T>);


TF_CALL_int32(REGISTER_KERNEL);
TF_CALL_int64(REGISTER_KERNEL);
#undef REGISTER_KERNEL