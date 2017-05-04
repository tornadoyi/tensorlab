#ifndef TENSORLAB_IMAGE_OPS_H_
#define TENSORLAB_IMAGE_OPS_H_


using namespace tensorflow;

#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/array_slice.h"




/// @defgroup user_ops User Ops
/// @{

/// TODO: add doc.
///
/// Arguments:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output_image tensor.
class AssignImage {
 public:
  AssignImage(const ::tensorflow::Scope& scope, ::tensorflow::Input src,
            ::tensorflow::Input dst, ::tensorflow::Input location);
  operator ::tensorflow::Output() const { return output_image; }
  operator ::tensorflow::Input() const { return output_image; }
  ::tensorflow::Node* node() const { return output_image.node(); }

  ::tensorflow::Output output_image;
};

/// Output a fact about factorials.
///
/// Arguments:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The fact tensor.
class Fact {
 public:
  Fact(const ::tensorflow::Scope& scope);
  operator ::tensorflow::Output() const { return fact; }
  operator ::tensorflow::Input() const { return fact; }
  ::tensorflow::Node* node() const { return fact.node(); }

  ::tensorflow::Output fact;
};

/// TODO: add doc.
///
/// Arguments:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The output_image tensor.
class FlatColor {
 public:
  FlatColor(const ::tensorflow::Scope& scope, ::tensorflow::Input src);
  operator ::tensorflow::Output() const { return output_image; }
  operator ::tensorflow::Input() const { return output_image; }
  ::tensorflow::Node* node() const { return output_image.node(); }

  ::tensorflow::Output output_image;
};

/// TODO: add doc.
///
/// Arguments:
/// * scope: A Scope object
///
/// Returns:
/// * `Output`: The pyramid_image tensor.
class Pyramid {
 public:
  /// Optional attribute setters for Pyramid
  struct Attrs {
    /// Defaults to 5
    Attrs MinSize(float x) {
      Attrs ret = *this;
      ret.min_size_ = x;
      return ret;
    }

    /// Defaults to 10
    Attrs Padding(int64 x) {
      Attrs ret = *this;
      ret.padding_ = x;
      return ret;
    }

    /// Defaults to "BILINEAR"
    Attrs Method(StringPiece x) {
      Attrs ret = *this;
      ret.method_ = x;
      return ret;
    }

    /// Defaults to false
    Attrs AlignCorners(bool x) {
      Attrs ret = *this;
      ret.align_corners_ = x;
      return ret;
    }

    float min_size_ = 5.0f;
    int64 padding_ = 10;
    StringPiece method_ = "BILINEAR";
    bool align_corners_ = false;
  };
  Pyramid(const ::tensorflow::Scope& scope, ::tensorflow::Input images,
        ::tensorflow::Input scale);
  Pyramid(const ::tensorflow::Scope& scope, ::tensorflow::Input images,
        ::tensorflow::Input scale, const Pyramid::Attrs& attrs);
  operator ::tensorflow::Output() const { return pyramid_image; }
  operator ::tensorflow::Input() const { return pyramid_image; }
  ::tensorflow::Node* node() const { return pyramid_image.node(); }

  static Attrs MinSize(float x) {
    return Attrs().MinSize(x);
  }
  static Attrs Padding(int64 x) {
    return Attrs().Padding(x);
  }
  static Attrs Method(StringPiece x) {
    return Attrs().Method(x);
  }
  static Attrs AlignCorners(bool x) {
    return Attrs().AlignCorners(x);
  }

  ::tensorflow::Output pyramid_image;
};

/// @}


#endif // #ifndef TENSORLAB_IMAGE_OPS_H_
