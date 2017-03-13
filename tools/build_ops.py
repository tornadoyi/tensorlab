import argparse
import os
import sys
import shutil
from utils import *

ROOT_PATH = os.path.dirname(os.path.realpath(sys.argv[0]))

OPS_PATH = os.path.join(ROOT_PATH, "../cc", "tensorflow-ext", "ops")
REG_OPS = os.path.join(ROOT_PATH, "../cc", "tensorflow-ext", "register")


header_module = "#ifndef TENSORLAB_{0}_H_\n" \
                "#define TENSORLAB_{0}_H_\n" \
                "\n\n" \
                "using namespace tensorflow;" \
                "\n\n" \
                "#include \"tensorflow/cc/framework/ops.h\"\n" \
                "#include \"tensorflow/cc/framework/scope.h\"\n" \
                "#include \"tensorflow/core/framework/tensor.h\"\n" \
                "#include \"tensorflow/core/framework/tensor_shape.h\"\n" \
                "#include \"tensorflow/core/framework/types.h\"\n" \
                "#include \"tensorflow/core/lib/gtl/array_slice.h\"\n" \
                "\n\n" \
                "{1}" \
                "\n\n" \
                "#endif // #ifndef TENSORLAB_{0}_H_\n"


source_module = "#include \"tensorflow/cc/ops/const_op.h\"\n" \
                "#include \"{0}.h\"\n" \
                "\n\n" \
                "{1}" \
                "\n\n"


def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorflow', type=str, help="tensorflow path")
    args = parser.parse_args()
    return args


def modify_content(file):
    filename = os.path.basename(file)
    split = filename.split(".")
    name = split[0]
    is_header = split[1] == 'h'

    with open(file, "r") as f:
        content = f.read()

    # pick content
    namespace_span = search_keyword(content, "namespace", "ops", "{")
    end = find_pair(content, namespace_span[1]-1, "{", "}")

    content = content[namespace_span[1] : end-1]

    if is_header:
        newcontent = header_module.format(name.upper(), content)
    else:
        newcontent = source_module.format(name, content)

    with open(file, "w") as f:
        f.write(newcontent)




def build_ops(tensorflow_path):
    # regenerate ops path
    shutil.rmtree(OPS_PATH)
    os.mkdir(OPS_PATH)

    cur_path = os.getcwd()
    os.chdir(tensorflow_path)
    user_ops_path = os.path.join(tensorflow_path, "tensorflow", "core", "user_ops")

    # clear user_ops
    remove_path_files(user_ops_path)

    ops_files = []
    for f in os.listdir(REG_OPS):
        srcf = os.path.join(REG_OPS, f)
        if not os.path.isfile(srcf): continue
        ops_files.append(f)


    for fname in ops_files:
        srcf = os.path.join(REG_OPS, f)
        dstf = os.path.join(user_ops_path, f)

        shutil.copy(srcf, dstf)
        os.system("bazel build --config=opt //tensorflow/cc:cc_ops")
        os.remove(dstf)

        srcf_cc = os.path.join(tensorflow_path, "bazel-genfiles/tensorflow/cc/ops/user_ops.cc")
        dstf_cc = os.path.join(OPS_PATH, fname)

        srcf_h = srcf_cc.replace(".cc", ".h")
        dstf_h = dstf_cc.replace(".cc", ".h")

        shutil.copy(srcf_cc, dstf_cc)
        shutil.copy(srcf_h, dstf_h)

        os.system("chmod 777 {0}".format(dstf_cc))
        os.system("chmod 777 {0}".format(dstf_h))

        modify_content(dstf_cc)
        modify_content(dstf_h)

    os.chdir(cur_path)



def main():
    args = parse()
    tensorflow_path = os.path.abspath(args.tensorflow)
    build_ops(tensorflow_path)


if __name__ == "__main__":
    main()