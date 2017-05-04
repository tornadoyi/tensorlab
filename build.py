import argparse
import os
import sys
import shutil

ROOT_PATH = os.path.dirname(os.path.realpath(sys.argv[0]))
BUILD_PATH = os.getcwd()
BUILD_CC_PATH = os.path.join(BUILD_PATH, "cmake_build")
BUILD_PY_PATH = os.path.join(BUILD_PATH, "python_build")

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["all", "c++", "python", "clean"], default='all', help="choice command")
    args = parser.parse_args()
    return args


def build_cc():
    cur_path = os.getcwd()

    # enter build cc path
    if not os.path.isdir(BUILD_CC_PATH): os.mkdir(BUILD_CC_PATH)
    os.chdir(BUILD_CC_PATH)

    # build tensorlab
    os.system('cmake {0}'.format(ROOT_PATH))
    os.system('make')

    # exit build cc path
    os.chdir(cur_path)

def build_py():
    cur_path = os.getcwd()

    # enter build py path
    if not os.path.isdir(BUILD_PY_PATH): os.mkdir(BUILD_PY_PATH)
    os.chdir(BUILD_PY_PATH)

    # copy python path
    shutil.copytree(os.path.join(ROOT_PATH, "tensorlab"), os.path.join(BUILD_PY_PATH, "tensorlab"))

    # copy setup file
    shutil.copy(os.path.join(ROOT_PATH, "setup.py"), BUILD_PY_PATH)

    # copy lib
    lib_path = os.path.join(BUILD_CC_PATH, "tensorlab/cc/tensorflow-ext/libtensorflow-ext.so")
    if not os.path.isfile(lib_path):
        raise Exception("{0} not exist".format(lib_path))
    else:
        shutil.copy(lib_path, "tensorlab")

    lib_path = os.path.join(BUILD_CC_PATH, "tensorlab/cc/tensorlab/libtensorlab.so")
    if not os.path.isfile(lib_path):
        raise Exception("{0} not exist".format(lib_path))
    else:
        shutil.copyfile(lib_path, "tensorlab/tensorlab.so")

    # build tensorlab
    os.system("python setup.py install")

    # exit build cc path
    os.chdir(cur_path)


def clean(path):
    # clear build path
    os.system("rm -rf {0}".format(path))


if __name__ == "__main__":
    args = parse()
    if args.command == "all":
        clean(BUILD_CC_PATH)
        clean(BUILD_PY_PATH)
        build_cc()
        build_py()

    elif args.command == "c++":
        clean(BUILD_CC_PATH)
        build_cc()

    elif args.command == "python":
        clean(BUILD_PY_PATH)
        build_py()

    elif args.command == "clean":
        clean(BUILD_CC_PATH)
        clean(BUILD_PY_PATH)

    else:
        print("Invalid command {0}".format(args.command))