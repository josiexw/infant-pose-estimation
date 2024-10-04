# --------------------------------------------------------
# Pose.gluon
# Copyright (c) 2018-present Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# --------------------------------------------------------

import os
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np


def find_in_path(name, path):
    """Find a file in a search path."""
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    # First check CUDAHOME
    cuda_home = os.environ.get('CUDAHOME') or os.environ.get('CUDA_HOME')
    if cuda_home is None:
        # Otherwise, search for CUDA in some common locations
        cuda_home = '/usr/local/cuda'
    
    if not os.path.exists(cuda_home):
        raise EnvironmentError('CUDA_HOME directory does not exist: ' + cuda_home)

    nvcc = os.path.join(cuda_home, 'bin', 'nvcc')
    if not os.path.exists(nvcc):
        raise EnvironmentError('NVCC not found in: ' + nvcc)

    # Set and return CUDA settings
    return {
        'home': cuda_home,
        'nvcc': nvcc,
        'include': os.path.join(cuda_home, 'include'),
        'lib64': os.path.join(cuda_home, 'lib64')
    }


CUDA = locate_cuda()


try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()


def customize_compiler_for_nvcc(self):
    """Customize how the dispatch to nvcc works."""
    self.src_extensions.append('.cu')

    super_compile = self._compile

    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == '.cu':
            self.set_executable('compiler_cxx', CUDA['nvcc'])  # Use compiler_cxx for nvcc
            postargs = extra_postargs['nvcc']
        else:
            postargs = extra_postargs['gcc']

        super_compile(obj, src, ext, cc_args, postargs, pp_opts)

    self._compile = _compile


class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


ext_modules = [
    Extension(
        "cpu_nms",
        ["cpu_nms.pyx"],
        extra_compile_args={'gcc': ["/W3", "/wd4244"],  # Disable warning for possible data loss
                            'nvcc': []},
        include_dirs=[numpy_include]
    ),
    Extension('gpu_nms',
        ['nms_kernel.cu', 'gpu_nms.pyx'],
        library_dirs=[CUDA['lib']],
        libraries=['cudart'],
        language='c++',
        runtime_library_dirs=[CUDA['lib']],
        extra_compile_args={'gcc': ["/W3", "/wd4244"],  # Disable warning for possible data loss
                            'nvcc': ['/arch=sm_61',
                                     '--ptxas-options=-v',
                                     '-c',
                                     '--compiler-options',
                                     '/Zi']},
        include_dirs=[numpy_include, CUDA['include']]
    ),
]

setup(
    name='nms',
    ext_modules=ext_modules,
    cmdclass={'build_ext': custom_build_ext},
)
