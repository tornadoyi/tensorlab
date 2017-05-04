//
// Created by Jason on 30/03/2017.
//

#ifndef TENSORLAB_PYRAMID_PLAN_H
#define TENSORLAB_PYRAMID_PLAN_H


#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/numeric_types.h"
#include "tensorflow/core/framework/tensor_types.h"


template <typename T, size_t UNUSED=0>
struct MakePyramidPlan
{
    // output_rects: point_y point_x height width
    void operator()(
            T input_height,
            T intput_width,
            T scale,
            T min_size,
            T padding,
            T& output_height,
            T& output_width,
            std::vector<std::tuple<float, float, float, float>>& output_rects);
};


#endif //PROJECT_PYRAMID_PLAN_OP_H
