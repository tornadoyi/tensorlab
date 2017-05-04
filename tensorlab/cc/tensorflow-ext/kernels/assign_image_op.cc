
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"
#include "image_tool.h"
#include "assign_image_op.h"
#include "support/algorithm.h"

using namespace tensorflow;

typedef Eigen::ThreadPoolDevice CPUDevice;


template <typename Device, typename T>
class AssignImageOp : public OpKernel
{
public:
    explicit AssignImageOp(OpKernelConstruction* context) : OpKernel(context)
    {

    }

    void Compute(OpKernelContext* context) override
    {
        const Tensor& src_tensor = context->input(0);
        const Tensor& dst_tensor = context->input(1);
        const Tensor& rects_tensor = context->input(2);
        const Tensor& indexes_tensor = context->input(3);


        OP_REQUIRES(context,
                    src_tensor.dims() == 4 &&
                    dst_tensor.dims() == 4 &&
                    _nd(src_tensor) == _nd(dst_tensor),
                    errors::InvalidArgument("image src and dst must be 4-dimensional ",
                                            "src: ", src_tensor.shape().DebugString(), " ",
                                            "dst: ", dst_tensor.shape().DebugString(), " "
                    ));


        OP_REQUIRES(context, rects_tensor.dims() == 2 &&
                    rects_tensor.dim_size(0) == src_tensor.dim_size(0) &&
                    rects_tensor.dim_size(1) == 4,
                    errors::InvalidArgument("rects_tensor must be 2-dimensional with N x (y1 x1 y2 x2)",
                                            rects_tensor.shape().DebugString()));


        OP_REQUIRES(context, indexes_tensor.dims() == 2 && indexes_tensor.dim_size(1) == 3,
                    errors::InvalidArgument("rects_tensor must be 2-dimensional with N x 3 (idx_src idx_dst _idx_rect)",
                                            indexes_tensor.shape().DebugString()));



        Tensor* output;
        OP_REQUIRES_OK(context, context->allocate_output(0, dst_tensor.shape(), &output));
        CHECK(output->CopyFrom(dst_tensor, dst_tensor.shape()));


        kernel::AssignImage<Device, T>()(context->eigen_device<Device>(),
                                         src_tensor.tensor<T, 4>(),
                                         rects_tensor.tensor<float, 2>(),
                                         indexes_tensor.tensor<int32, 2>(),
                                         output->tensor<float, 4>());

    }

};



namespace kernel
{
    template <typename T>
    struct AssignImage<CPUDevice, T>
    {
        void operator()(const CPUDevice& d,
                        typename TTypes<T, 4>::ConstTensor src_data,
                        typename TTypes<float, 2>::ConstTensor rects,
                        typename TTypes<int32, 2>::ConstTensor indexes,
                        typename TTypes<float, 4>::Tensor output_data) try
        {
            auto b_in = _nb(src_data);
            auto b_out = _nb(output_data);
            auto b_rect = rects.dimension(0);
            auto channel = _nd(output_data);

            auto r_out = _nr(output_data);
            auto c_out = _nc(output_data);

            auto r_in = _nr(src_data);
            auto c_in = _nc(src_data);


            auto index_count = indexes.dimension(0);

            for (int64 i = 0; i < index_count; ++i)
            {
                auto idx_src = indexes(i, 0);
                auto idx_dst = indexes(i, 1);
                auto idx_rect = indexes(i, 2);


                if(idx_src <0 || idx_src >= b_in ||
                        idx_rect < 0 || idx_rect >= b_rect ||
                        idx_dst < 0 || idx_dst >= b_out)
                {
                    LOG(ERROR) << "out of range "
                               << "idx_src: " << idx_src << "<" << b_in << " "
                               << "idx_dst: " << idx_dst << "<" << idx_dst << " "
                               << "idx_rect: " << idx_rect << "<" << b_rect << " ";

                    continue;
                }

                auto st_p_y = rects(idx_rect, 0);
                auto st_p_x = rects(idx_rect, 1);
                auto ed_p_y = rects(idx_rect, 2);
                auto ed_p_x = rects(idx_rect, 3);

                auto st_y = (int32)clip(st_p_y*r_out-1, 0.f, (float)r_out-1.f);
                auto st_x = (int32)clip(st_p_x*c_out-1, 0.f, (float)c_out-1.f);
                auto ed_y = (int32)clip(ed_p_y*r_out-1, 0.f, (float)r_out-1.f);
                auto ed_x = (int32)clip(ed_p_x*c_out-1, 0.f, (float)c_out-1.f);
                auto total_row = ed_y - st_y + 1;
                auto total_col = ed_x - st_x + 1;

                if(st_y < 0 || st_y > ed_y || ed_y >= r_out ||
                   st_x < 0 || st_x > ed_x || ed_x >= c_out)
                {
                    LOG(ERROR) << "out of range "
                               << "st_y: " << st_y << " < " << "ed_y: " << ed_y << " < " << r_out
                               << "st_x: " << st_x << " < " << "ed_x: " << ed_x << " < " << c_out;
                    continue;
                }


                float scale_r = (float)r_in / (float)total_row;
                float scale_c = (float)c_in / (float)total_col;

                //LOG(INFO) << st_y << " " << st_x << " " <<totol_row << " " << totol_col;
                //LOG(INFO) << scale_r << " " << scale_c;

                for(int64 y=st_y; y <= ed_y; ++y)
                {
                    // map output location-y to input-y
                    auto in_y = (y-st_y) * scale_r;
                    auto lower_in_y = static_cast<int64>(in_y);
                    auto upper_in_y = std::min(lower_in_y + 1, r_in - 1);
                    auto lerp_in_y = in_y - lower_in_y;

                    for(int64 x=st_x; x <= ed_x; ++x)
                    {
                        // map output location-x to input-x
                        auto in_x = (x-st_x) * scale_c;
                        auto lower_in_x = static_cast<int64>(in_x);
                        auto upper_in_x = std::min(lower_in_x + 1, c_in - 1);
                        auto lerp_in_x = in_x - lower_in_x;

                        //LOG(INFO)<<"lower_in_x: " << lower_in_x << " " <<"upper_in_x: " << upper_in_x << " ";


                        for(int64 d=0; d < channel; ++d)
                        {
                            auto tl = (float)src_data(idx_src, lower_in_y, lower_in_x, d);
                            auto tr = (float)src_data(idx_src, lower_in_y, upper_in_x, d);
                            auto bl = (float)src_data(idx_src, upper_in_y, lower_in_x, d);
                            auto br = (float)src_data(idx_src, upper_in_y, upper_in_x, d);


                            // calculate lerp
                            const auto top = tl + (tr - tl) * lerp_in_x;
                            const auto bottom = bl + (br - bl) * lerp_in_x;
                            auto lerp = top + (bottom - top) * lerp_in_y;
                            output_data(idx_dst, y, x, d) = lerp;

                            /*
                            LOG(INFO)<<"y: " << y << " " << "x: " << x << " "
                                     <<"tl: " << tl << " "
                                     <<"tr: " << tr << " "
                                     <<"bl: " << bl << " "
                                     <<"br: " << br << " "
                                     <<"lerp_in_x: " << lerp_in_x << " "
                                     <<"lerp_in_y: " << lerp_in_y << " ";
                            LOG(INFO) << "top: " << top << " bottom: " << bottom << " lerp: "<<lerp;
                            */

                        }
                    }
                }// for(int64 y=0; y < r_out; ++y)

            }// for (int64 b = 0; b < batch; ++b)

        }
        catch(std::exception& e)
        {
            LOG(FATAL) << e.what();
        }
    };


#define DEFINE_CPU_SPECS(T)                     \
    template struct AssignImage<CPUDevice, T>; \

TF_CALL_REAL_NUMBER_TYPES(DEFINE_CPU_SPECS);

#undef DEFINE_CPU_SPECS
}


#define REGISTER_KERNEL(T)                            \
  REGISTER_KERNEL_BUILDER(Name("AssignImage")      \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T"), \
                          AssignImageOp<CPUDevice, T>);



TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL




