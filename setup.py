# ----------------------------------------------------------------------------
# Copyright 2017 Happy elements Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
import os
import sys
import shutil
from setuptools import setup, find_packages, Command


# Define version information
VERSION = '0.1'
FULLVERSION = VERSION
ROOT_PATH = os.path.dirname(os.path.realpath(sys.argv[0]))
BUILD_PATH = os.getcwd()
PYTHON_SRC_PATH = os.path.join(ROOT_PATH, "python")
LIB_SRC_PATH = os.path.join(BUILD_PATH, "tensorlab", "libtensorlab.so")
LIB_DST_PATH = os.path.join(PYTHON_SRC_PATH, "libtensorlab.so")

# clear build path
os.system("rm -rf *")

# build tensorlab
os.system('cmake ..')
os.system('make')

# copy lib
if not os.path.isfile(LIB_SRC_PATH):
    exit()
shutil.copy(LIB_SRC_PATH, LIB_DST_PATH)

# build python

setup(name='tensorlab',
    version=VERSION,
    description="extension for python grammar",
    #long_description=open('README.md').read(),
    author='yi gu',
    author_email='390512308@qq.com',
    license='License :: OSI Approved :: Apache Software License',
    packages=['tensorlab'],
    package_dir={'tensorlab': PYTHON_SRC_PATH},
    package_data={'tensorlab': ['*.so']},
    zip_safe=False,
    classifiers=['Development Status :: 3 - Alpha',
               'Environment :: Console',
               'Environment :: Console :: Curses',
               'Environment :: Web Environment',
               'Intended Audience :: End Users/Desktop',
               'Intended Audience :: Developers',
               'Intended Audience :: Science/Research',
               'License :: OSI Approved :: Apache Software License',
               'Operating System :: POSIX',
               'Operating System :: MacOS :: MacOS X',
               'Programming Language :: Python :: 2',
               'Programming Language :: Python :: 3',
               'Topic :: Scientific/Engineering :: ' +
               'Artificial Intelligence',
               'Topic :: Scientific/Engineering :: Information Analysis',
               'Topic :: System :: Distributed Computing'])


# delete lib
os.remove(LIB_DST_PATH)