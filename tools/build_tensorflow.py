import argparse
import os
import sys
import shutil

build_target_code = "cc_binary(\n" \
                    "   name = \"libtensorflow.so\",\n" \
                    "   linkshared = 1,\n" \
                    "   deps = [\n" \
                    "       \"//tensorflow/c:c_api\",\n" \
                    "       \"//tensorflow/cc:client_session\",\n" \
                    "       \"//tensorflow/cc:cc_ops\",\n" \
                    "       \"//tensorflow/core:tensorflow\",\n" \
                    "   ],\n" \
                    ")\n"

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["all", "config", "build"], default='all', help="choice command")
    parser.add_argument('--tensorflow', type=str, help="tensorflow path")
    parser.add_argument('--cuda', type=str2bool, default=False)
    args = parser.parse_args()
    return args


def find_build_target(content, target, attr):
    idx_target = -1
    idx_target_end = -1
    while True:
        idx_target = content.find(target, idx_target+1)
        if idx_target == -1: break
        idx_attr = content.find(attr, idx_target)
        if idx_attr == -1: continue

        lbracket = content.find("(", idx_target)
        if lbracket == -1: break
        count = 1
        for i in xrange(lbracket+1, len(content)):
            if content[i] == '(': count += 1
            elif content[i] == ')': count -= 1
            else: continue
            if count == 0:
                idx_target_end = i
                break
        if idx_target_end <= idx_attr: continue
        else: break

    return idx_target, idx_target_end



def modify_build_target(tensorflow_path):
    build_file_path = os.path.join(tensorflow_path, "tensorflow", "BUILD")
    with open(build_file_path, "r") as f:
        content = f.read()

    st, ed = find_build_target(content, 'cc_binary', "libtensorflow.so")
    newcontent = None
    if st == -1:
        content = content + "\n" + build_target_code
    else:
        newcontent = content[0:st]
        newcontent += content[ed+1:len(content)]
        newcontent += "\n" + build_target_code
        #print(content[st:ed+1])

    with open(build_file_path, "w") as f:
        f.write(newcontent)


def configure_tensorflow(tensorflow_path, config=True, build=True):
    cur_path = os.getcwd()
    os.chdir(tensorflow_path)
    if config: os.system("./configure")
    os.chdir(cur_path)

def build_tensorflow(tensorflow_path, cuda):
    cur_path = os.getcwd()

    os.chdir(tensorflow_path)
    if cuda:
        os.system("bazel build --config=opt --config=cuda //tensorflow:libtensorflow.so")
    else:
        os.system("bazel build --config=opt //tensorflow:libtensorflow.so")

    os.system("sudo cp bazel-bin/tensorflow/libtensorflow.so /usr/local/lib")
    os.chdir(cur_path)



def main():
    args = parse()
    tensorflow_path = os.path.abspath(args.tensorflow)
    if args.command == "all":
        modify_build_target(tensorflow_path)
        configure_tensorflow(tensorflow_path)
        build_tensorflow(tensorflow_path, args.cuda)

    elif args.command == "config":
        configure_tensorflow(tensorflow_path)

    else:
        modify_build_target(tensorflow_path)
        build_tensorflow(tensorflow_path, args.cuda)


if __name__ == "__main__":
    main()