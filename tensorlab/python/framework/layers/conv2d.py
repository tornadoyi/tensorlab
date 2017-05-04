from __future__ import division
from ..tensor import Tensor
import tensorflow as tf
import numpy as np
from .. import utils


# reference by tensorflow/core/framework/common_shape_fns.cc
# switch (padding_type) {
#   case Padding::VALID:
#       *output_size = (input_size - filter_size + stride) / stride;
#       *padding_before = *padding_after = 0;
#   case Padding::SAME:
#       *output_size = (input_size + stride - 1) / stride;
#       const int64 padding_needed = std::max(0LL, (*output_size - 1) * stride + filter_size - input_size);
#       *padding_before = padding_needed / 2;
#       *padding_after = padding_needed - *padding_before;


class conv2d(Tensor):
    def __init__(self, *args, **kwargs):
        Tensor.__init__(self, tf.nn.conv2d, *args, **kwargs)


    # [filter_height, filter_width, in_channels, out_channels]
    @property
    def filter(self): return utils.shape(self.arg_attr('filter'))

    # [1, stride, stride, 1]
    @property
    def strides(self): return self.arg_attr('strides')

    @property
    def padding(self): return self.arg_attr('padding')


    # input_shape: 1-D of length 2 vector, HW format
    def padding_value(self, input_shape):
        assert np.shape(input_shape) == (2,)

        filter = self.filter
        strides = self.strides
        padding = self.padding

        if padding == 'VALID':
            return tf.constant((0,0), dtype=tf.int32)

        else:
            output = (input_shape + [strides[1]-1, strides[2]-1]) / [strides[1], strides[2]]
            output = tf.cast(output, tf.int32)
            padding_needed = (output - [1, 1]) * [strides[1], strides[2]] + [filter[0], filter[1]] - input_shape
            padding_needed = tf.maximum(0, padding_needed) / 2

            return tf.cast(padding_needed, tf.int32)



    # input_shape: Tensor 1-D of length 4 vector, NHWC format
    def output_shape(self, input_shape):
        assert utils.shape(input_shape) == (4,)

        filter = self.filter
        strides = self.strides
        padding = self.padding

        if padding == 'VALID':
            output = (input_shape + [0, strides[1]-filter[0], strides[2]-filter[1], 0]) / [1, strides[1], strides[2], 1]

        else:
            output = (input_shape + [0, strides[1]-1, strides[2]-1, 0]) / [1, strides[1], strides[2], 1]

        output = tf.cast(output, tf.int32)
        return output * [1,1,1,0] + [0,0,0,filter[3]]


