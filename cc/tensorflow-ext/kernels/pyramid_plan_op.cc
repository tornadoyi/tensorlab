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
        const Tensor& scale_tensor = context->input(1);


        OP_REQUIRES(context, size_tensor.dims() == 1 && size_tensor.dim_size(0) == 2,
                    errors::InvalidArgument("size must be 1-dimensional with (height, width)",
                                            size_tensor.shape().DebugString()));

        OP_REQUIRES(context, TensorShapeUtils::IsScalar(scale_tensor.shape()),
                    errors::InvalidArgument("scale must be scalar",
                                            scale_tensor.DebugString()));

        auto scale = scale_tensor.flat<T>()(0);

        auto input_size = size_tensor.vec<T>();
        T input_r = input_size(0);
        T input_c = input_size(1);

        std::vector<std::tuple<float, float, float, float>> rects;
        T output_width, output_height;
        MakePyramidPlan<T>()(input_r, input_c, scale, min_size, padding, output_height, output_width, rects);
        //LOG(INFO) << output_height << " " << output_width;

        Tensor* output_size_tensor;
        OP_REQUIRES_OK(context, context->allocate_output(
                0,
                TensorShape({2}),
                &output_size_tensor));

        Tensor* rect_tensor;
        OP_REQUIRES_OK(context, context->allocate_output(
                1,
                TensorShape({int64(rects.size()), 4}),
                &rect_tensor));

        auto p_output_size = output_size_tensor->vec<int32>();
        p_output_size(0) = output_height;
        p_output_size(1) = output_width;

        typename TTypes<float, 2>::Tensor p_rect = rect_tensor->tensor<float, 2>();
        for(size_t i=0; i<rects.size(); ++i)
        {
            auto& rect = rects[i];
            p_rect(i, 0) = std::get<0>(rect);
            p_rect(i, 1) = std::get<1>(rect);
            p_rect(i, 2) = std::get<2>(rect);
            p_rect(i, 3) = std::get<3>(rect);
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
            std::vector<std::tuple<float, float, float, float>>& output_rects)
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

        float max_row_index = (float)(output_height - 1);
        float max_col_index = (float)(output_width - 1);

        T y = 0;
        size_t i = 0;
        while(y < height)
        {
            auto& size = pyramid[i];
            r = size.first; c = size.second;

            std::tuple<float, float, float, float> rect(
                    (float)y / max_row_index,
                    0.f,
                    (float)(y+r-1) / max_row_index,
                    (float)(c-1) / max_col_index);

            /*
            LOG(INFO)<< "(" << r << ", " << c << ") "
                    << std::get<0>(rect) << " "
                    << std::get<1>(rect) << " "
                    << std::get<2>(rect) << " "
                    << std::get<3>(rect) << " ";
                    */

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

            std::tuple<float, float, float, float> rect(
                    (float)tl_y / max_row_index,
                    (float)tl_x / max_col_index,
                    (float)br_y / max_row_index,
                    (float)br_x / max_col_index);
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