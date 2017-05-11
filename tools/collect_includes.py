import argparse
import os
import sys
import shutil

ROOT_PATH = os.path.dirname(os.path.realpath(sys.argv[0]))

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorflow', required=True, type=str, help="tensorflow path")
    args = parser.parse_args()
    return args


def copy_directory(src, dst):
    if not os.path.isdir(dst):
        shutil.copytree(src, dst)
        return

    for f in os.listdir(src):
        srcf = os.path.join(src, f)
        dstf = os.path.join(dst, f)

        # file then copy to dst
        if os.path.isfile(srcf):
            shutil.copy(srcf, dstf)
        else:
            if not os.path.exists(dstf):
                shutil.copytree(srcf, dstf)
            else:
                copy_directory(srcf, dstf)



def collect_includes(tensorflow_path):
    dst_include_path = os.path.realpath(os.path.join(ROOT_PATH, "../tensorlab/cc/include"))
    if os.path.isdir(dst_include_path): shutil.rmtree(dst_include_path)
    os.mkdir(dst_include_path)

    copy_directory(os.path.join(tensorflow_path, "tensorflow"), os.path.join(dst_include_path, "tensorflow"))
    copy_directory(os.path.join(tensorflow_path, "bazel-genfiles/tensorflow"), os.path.join(dst_include_path, "tensorflow"))


def main():
    args = parse()
    tensorflow_path = os.path.abspath(args.tensorflow)
    collect_includes(tensorflow_path)


if __name__ == "__main__":
    main()