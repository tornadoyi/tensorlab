#include "tensorflow/cc/ops/const_op.h"
#include "image_ops.h"




AssignImage::AssignImage(const ::tensorflow::Scope& scope, ::tensorflow::Input
                         src, ::tensorflow::Input dst, ::tensorflow::Input
                         location) {
  if (!scope.ok()) return;
  auto _src = ::tensorflow::ops::AsNodeOut(scope, src);
  if (!scope.ok()) return;
  auto _dst = ::tensorflow::ops::AsNodeOut(scope, dst);
  if (!scope.ok()) return;
  auto _location = ::tensorflow::ops::AsNodeOut(scope, location);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("AssignImage");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "AssignImage")
                     .Input(_src)
                     .Input(_dst)
                     .Input(_location)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  this->output_image = Output(ret, 0);
}

FlatColor::FlatColor(const ::tensorflow::Scope& scope, ::tensorflow::Input src) {
  if (!scope.ok()) return;
  auto _src = ::tensorflow::ops::AsNodeOut(scope, src);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("FlatColor");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "FlatColor")
                     .Input(_src)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  this->output_image = Output(ret, 0);
}

Pyramid::Pyramid(const ::tensorflow::Scope& scope, ::tensorflow::Input images,
                 ::tensorflow::Input scale, const Pyramid::Attrs& attrs) {
  if (!scope.ok()) return;
  auto _images = ::tensorflow::ops::AsNodeOut(scope, images);
  if (!scope.ok()) return;
  auto _scale = ::tensorflow::ops::AsNodeOut(scope, scale);
  if (!scope.ok()) return;
  ::tensorflow::Node* ret;
  const auto unique_name = scope.GetUniqueNameForOp("Pyramid");
  auto builder = ::tensorflow::NodeBuilder(unique_name, "Pyramid")
                     .Input(_images)
                     .Input(_scale)
                     .Attr("min_size", attrs.min_size_)
                     .Attr("padding", attrs.padding_)
                     .Attr("method", attrs.method_)
                     .Attr("align_corners", attrs.align_corners_)
  ;
  scope.UpdateBuilder(&builder);
  scope.UpdateStatus(builder.Finalize(scope.graph(), &ret));
  if (!scope.ok()) return;
  this->pyramid_image = Output(ret, 0);
}

Pyramid::Pyramid(const ::tensorflow::Scope& scope, ::tensorflow::Input images,
                 ::tensorflow::Input scale)
  : Pyramid(scope, images, scale, Pyramid::Attrs()) {}

/// 

